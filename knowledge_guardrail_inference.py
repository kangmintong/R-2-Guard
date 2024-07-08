import argparse
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc
from data_loading import load_data
from utils import load_scores_given_path, load_field_name, construct_weight_graph, evaluate_asr, save_to_jsonl
from knowledge_guardrail_model import knowledge_inference_model
import multiprocessing
import time

def knowledge_inference_partition(model, scores_all_one_instance, args):
    if not args.AC_inference:
        for scores in scores_all_one_instance:
            scores.append(.5)
        prob_unsafe = model(scores_all_one_instance, args.parallel_reasoning, args.reasoning_processes)
    else:
        scores_graph, weight_graph = construct_weight_graph(scores_all_one_instance, model.weights_all)
        prob_unsafe = model(scores_graph, args.parallel_reasoning, args.reasoning_processes, AC_inference=True, weight_graph=weight_graph)
    return prob_unsafe

def compute_one_instance(process_id, output, idx_list, scores_all_models, model, args):
    probs_unsafe = []
    for idx in idx_list:
        if args.ensemble_max:
            probs_unsafe.append(max(scores_all_models[i][fil][idx] for i in range(len(scores_all_models)) for fil in load_field_name(args.knowledge_model_name[i])))
            continue
        elif args.ensemble_avg:
            probs_unsafe.append(sum(max(scores_all_models[i][fil][idx] for fil in load_field_name(args.knowledge_model_name[i])) for i in range(len(scores_all_models)))/len(scores_all_models))
            continue

        scores_all_one_instance = []
        for i in range(len(scores_all_models)):
            scores_all_one_instance.append([scores_all_models[i][fil][idx] for fil in load_field_name(args.knowledge_model_name[i])])
        prob_unsafe = knowledge_inference_partition(model, scores_all_one_instance, args)
        probs_unsafe.append(prob_unsafe)
    output.put([process_id, probs_unsafe])

def run_knowledge_guardrail(args):
    num_models = len(args.knowledge_model_name)
    model_list = args.knowledge_model_name
    scores_all_models = []
    dim_list = []
    for i in range(num_models):
        score_path = f'./cache/{model_list[i]}_{args.dataset}_scores{args.advbench_suffix}.json'
        scores_all_models.append(load_scores_given_path(score_path))
        dim_list.append(len(load_field_name(model_list[i])))

    instances, categories = load_data(args.dataset, args)

    if args.load_knowledge_weights:
        model = knowledge_inference_model(scores_all_models, dim_list=dim_list,load_knowledge_weight=True, args=args)
        model.weight_init_1()
    else:
        model = knowledge_inference_model(scores_all_models, dim_list=dim_list)
        model.weight_init(weight=args.init_weight, agg_weights=[0.05,0.05,0.9]) # 0.65,0.05,0.3 # beavertail: 0.05 0.2 0.75
        # model.weight_init_1()
        # model.weight_init_llamaguard_twin()

    num_processes = args.num_processes
    processes = []
    output = multiprocessing.Queue()

    st_time = time.time()

    start = 0
    batch_size = len(instances) // num_processes + 1
    for i in range(num_processes):
        process = multiprocessing.Process(target=compute_one_instance, args=(i, output, list(range(start, min(start + batch_size, len(instances)))), scores_all_models, model, args))
        start += batch_size
        process.start()
        processes.append(process)

    probs_unsafe_dict = {}
    probs_unsafe = []
    for i in range(num_processes):
        result = output.get()
        probs_unsafe_dict[result[0]] = result[1]
    for i in range(num_processes):
        probs_unsafe = probs_unsafe + probs_unsafe_dict[i]

    for process in processes:
        process.join()

    end_time = time.time()
    print(f'Runtime: {end_time - st_time} seconds')

    if args.dataset=='dro':
        num = 400
        probs_unsafe, categories = probs_unsafe[:num], categories[:num]
        for i in range(len(categories)):
            categories[i] = 1 - categories[i]

    precision, recall, thresholds = precision_recall_curve(categories, probs_unsafe)
    auprc = auc(recall, precision)
    print(f'AUPRC: {auprc}')

    if 'adv' in args.dataset:
        racc = evaluate_asr(categories, probs_unsafe, thresh=0.5)
        print(f'Robust accuracy: {racc}')

    if 'mod_' in args.dataset:
        acc = evaluate_asr(categories, probs_unsafe, thresh=0.5,dataset=args.dataset)
        print(f'False negative rate: {1.-acc}')

    if args.save_probs_unsafe:
        save_to_jsonl(instances, probs_unsafe, categories, args)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Arguments for running knowledge guardrail")
    parser.add_argument('--knowledge_model_name', nargs='+', help="name of used knowledge model")
    parser.add_argument('--dataset', type=str, help="dataset", choices=['openaimod', 'toxicchat', 'toxicchat_train', 'advbench_string', 'advbench_behaviour', 'advbench_behaviour_hotpot', 'advbench_string_hotpot', 'dro', 'xstest', 'overkill', 'ours', 'beavertail','mod_hate','mod_sex','mod_harassment','mod_selfharm','mod_violence'],default='openaimod')
    parser.add_argument('--load_knowledge_weights', action='store_true')
    parser.add_argument('--parallel_reasoning', action='store_true')
    parser.add_argument('--training_dataset', type=str, choices=['pseudo', 'toxicchat_train'])
    parser.add_argument('--num_processes', type=int, default=300)
    parser.add_argument('--reasoning_processes', type=int, default=300)
    parser.add_argument('--advbench_suffix', required=False, type=str, default='', help="adversarial suffix for AdvBench")
    parser.add_argument('--AC_inference', action='store_true')
    parser.add_argument('--ensemble_avg', action='store_true')
    parser.add_argument('--ensemble_max', action='store_true')
    parser.add_argument('--save_probs_unsafe', action='store_true')
    parser.add_argument('--init_weight', type=float, default=66.0)
    args = parser.parse_args()

    run_knowledge_guardrail(args)