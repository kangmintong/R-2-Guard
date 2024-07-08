import argparse
from tqdm import tqdm
import json
from knowledge_guardrail_model import knowledge_inference_model
import numpy as np
import random
import multiprocessing
from data_loading import construct_pseudo_training_set, sample_real
from utils import load_field_name, print_out_knowledge_weights

def knowledge_train(model, scores_cur, label, psudo):
    for scores in scores_cur:
        scores.append(.5)
    if psudo:
        gradient = model.pseudo_learning(scores_cur, label)
    else:
        gradient = model.gradient_to_weights(scores_cur, label)
    return gradient
def get_gradient_one_instance(output, model, scores_cur, label, pseudo):
    gradient, loss = knowledge_train(model, scores_cur, label, pseudo)
    output.put([gradient, loss._value])
def knowledge_weight_learning(args):
    data_size = args.data_size
    lr1 = args.lr1
    lr2 = args.lr2
    batch_size = args.batch_size
    num_batch = data_size // batch_size + 1
    pseudo = 'pseudo' in args.dataset
    num_models = len(args.knowledge_model_name)
    model_list = args.knowledge_model_name
    dim_list = []
    for i in range(num_models):
        dim_list.append(len(load_field_name(model_list[i])))
    if pseudo:
        scores_all, labels = construct_pseudo_training_set(data_size=data_size, dim_list=dim_list)
    else:
        scores_all, labels = sample_real(data_size=args.data_size, model_names=model_list, dataset=args.dataset, args=args)
    model = knowledge_inference_model(dim_list=dim_list)
    model.weight_init(weight=1.0, agg_weights=[0.5, 0.25, 0.25])


    mavg_gradients_sqr = []
    for i in range(num_models):
        mavg_gradients_sqr.append(np.zeros_like(model.weights_all[i]))
    mavg_gradients_sqr.append(np.zeros_like(model.agg_weights))
    mavg_grad_pseudo = 0.0

    for it in tqdm(range(args.epochs)):
        start = -batch_size
        for batch_idx in tqdm(range(num_batch + 1)):
            start += batch_size
            if start >= data_size:
                break
            end = min(start + batch_size, data_size)
            processes = []
            output = multiprocessing.Queue()
            for i in range(end - start):
                scores_cur = scores_all[start + i]
                label = labels[start + i]
                process = multiprocessing.Process(target=get_gradient_one_instance, args=(output, model, scores_cur, label, pseudo))
                process.start()
                processes.append(process)
            cache_gradients = []
            losses = []
            for i in range(end - start):
                result = output.get()
                cache_gradients.append(result[0])
                losses.append(result[1])
            for process in processes:
                process.join()
            print(f'loss: {sum(losses) / len(losses)}')
            model.loss_records = []
            if not pseudo:
                for k in range(num_models):
                    cg = []
                    for j in range(end - start):
                        cg.append(cache_gradients[j][k])
                    cg = np.array(cg)
                    sqr_cg = np.mean(cg * cg, axis=0)
                    mavg_gradients_sqr[k] = 0.9 * mavg_gradients_sqr[k] + 0.1 * sqr_cg
                    avg_gradients = np.mean(cg, axis=0)
                    model.weights_all[k] -= lr1 * avg_gradients / (mavg_gradients_sqr[k] + 1e-8)
                cg = []
                for j in range(end - start):
                    cg.append(cache_gradients[j][-1])
                cg = np.array(cg)
                sqr_cg = np.mean(cg * cg, axis=0)
                mavg_gradients_sqr[-1] = 0.9 * mavg_gradients_sqr[-1] + 0.1 * sqr_cg
                avg_gradients = np.mean(cg, axis=0)
                model.agg_weights -= lr2 * avg_gradients / (mavg_gradients_sqr[-1] + 1e-8)

                model.agg_weights = np.array(model.agg_weights)
                model.agg_weights = model.agg_weights / sum(model.agg_weights)
                model.agg_weights[model.agg_weights<0.02] = 0.02
            else:
                cache_gradients = np.array(cache_gradients)
                sqr_grad = np.mean(cache_gradients * cache_gradients, axis=0)
                avg_grad = np.mean(cache_gradients, axis=0)
                mavg_grad_pseudo = 0.9 * mavg_grad_pseudo + 0.1 * sqr_grad
                delta_weight = - lr1 / (mavg_grad_pseudo + 1e-10) * avg_grad
                model.apply_delta_weight(delta_weight)
            print_out_knowledge_weights(model)
        if args.epochs==args.epochs//2+1:
            lr1 *= 0.1
            lr2 *= 0.1

    for i in range(num_models):
        np.save(f'./cache/knowledge_weights_{args.dataset}_component_{i}', model.weights_all[i])
    np.save(f'./cache/knowledge_weights_{args.dataset}_agg_weights', model.agg_weights)
    print(f'Finish saving knowledge weights')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Arguments for optimization knowledge weights")
    parser.add_argument('--knowledge_model_name', '--names-list', nargs='+', help="name of used knowledge model")
    parser.add_argument('--dataset', type=str, help="dataset", choices=['pseudo', 'toxicchat_train'])
    parser.add_argument('--data_size', required=False, type=int, default=200)
    parser.add_argument('--pos_ratio', required=False, type=float, default=0.3)
    parser.add_argument('--batch_size', required=False, type=int, default=10)
    parser.add_argument('--epochs', required=False, type=int, default=5)
    parser.add_argument('--lr1', required=False, type=float, default=1e-3)
    parser.add_argument('--lr2', required=False, type=float, default=1e-3)
    args = parser.parse_args()

    knowledge_weight_learning(args)

