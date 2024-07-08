import torch.nn as nn
import autograd.numpy as gradnp
from autograd import grad
import numpy as np
import multiprocessing


def int2assignment(idx, n):
    assign = []
    for k in range(n):
        assign.append(idx%2)
        idx = idx // 2
    assign.reverse()
    return assign


def knowledge_inference_prob(scores, weights):
    n = len(scores)
    nomi, denomi = 0., 0.
    for idx in range(2**n):
        assignments = int2assignment(idx, n)
        cur = 1.0
        for k in range(len(scores)):
            assi = assignments[k]
            cur *= (1 - scores[k]) * (1 - assi) + scores[k] * assi
            if assi==1 and assignments[-1]==0:
                cur *= 1
            else:
                if k < len(weights):
                    cur *= weights[k]
        denomi += cur
        if assignments[-1]==1:
            nomi += cur
    return nomi / denomi


class knowledge_inference_model(nn.Module):
    def __init__(self, scores_all_models=None, dim_list=None, weights_all=None, agg_weights=None, alpha=1.0, load_knowledge_weight=False, args=None):
        super(knowledge_inference_model, self).__init__()
        self.weights_all = weights_all
        self.agg_weights = agg_weights
        self.loss_records = []
        self.alpha = alpha

        if scores_all_models:
            self.num_models = len(scores_all_models)
            self.list_dim = []
            for i in range(self.num_models):
                self.list_dim.append(len(scores_all_models[i]))
        if dim_list:
            self.num_models = len(dim_list)
            self.dim_list = dim_list
            self.weight_init()
        else:
            raise ValueError("Provide dim_list!")


        if load_knowledge_weight:
            weights_all_load = []
            for i in range(self.num_models):
                weights = np.load(f'./cache/knowledge_weights_{args.training_dataset}_component_{i}.npy')
                weights_all_load.append(weights)
            self.weights_all = weights_all_load
            print(f'loading knowledge weights')
            print(self.weights_all)
            self.agg_weights = np.load(f'./cache/knowledge_weights_{args.training_dataset}_agg_weights.npy')
            print(self.agg_weights)

    def weight_init(self, weight=66.0, agg_weights = [0.5, 0.25, 0.25]):
        weight1 = weight
        weights_all = []
        for i in range(self.num_models):
            weights_all.append(np.ones((self.dim_list[i] + 1, self.dim_list[i] + 1)))
        for i in range(self.num_models):
            weights_all[i][:-1, -1] = weight1
        self.weights_all = weights_all
        self.agg_weights = agg_weights

    def weight_init_1(self):
        weight1 = 66.0
        weight2 = 66.0
        # additional implication rules
        self.weights_all[0][2, 0] = weight1
        self.weights_all[0][5, 1] = weight1
        self.weights_all[0][6, 10] = weight1
        self.weights_all[0][9, 2] = weight1
        try:
            self.weights_all[1][1, 0] = weight2
            # self.weights_all[2][1, 0] = weight2
            self.weights_all[2][3, 0] = weight2
            self.weights_all[2][3, 4] = weight2
        except:
            pass

    def weight_init_llamaguard_twin(self):
        weight1 = 66.0
        weight2 = 9.5
        # additional implication rules
        # self.weights_all[0][2, 0] = weight1
        # self.weights_all[0][5, 1] = weight1
        # self.weights_all[0][6, 10] = weight1
        # self.weights_all[0][9, 2] = weight1
        try:
            self.weights_all[1][1, 0] = weight2
            self.weights_all[1][3, 2] = weight2 + 1.0
            self.weights_all[1][4, 2] = weight2 + 1.0
            self.weights_all[1][6, 0] = weight2 + 1.0
            self.weights_all[1][6, 1] = weight2
            # self.weights_all[1][6, 2] = weight2
            self.weights_all[2][1, 0] = 100.0
            # self.weights_all[2][3, 0] = weight2
            # self.weights_all[2][3, 4] = weight2
        except:
            pass

    def knowledge_inference_prob(self, scores, weights, parallel=False, num_process=0):
        if parallel:
            return self.knowledge_inference_prob_parallel(scores, weights, num_process=num_process)
        n = len(scores)
        nomi, denomi = 0., 0.
        for idx in range(2 ** n):
            assignments = int2assignment(idx, n)

            cur = 1.0
            for k in range(n):
                assi = assignments[k]
                cur *= (1 - scores[k]) * (1 - assi) + scores[k] * assi
                for k2 in range(n):
                    if not(assignments[k]==1 and assignments[k2]==0):
                        cur *= weights[k][k2]
                    # if assignments[k]==1 and assignments[k2]==0:
                    #     cur *= weights[k][k2]
                # if assi == 1 and assignments[-1] == 0:
                #     cur *= 1
                # else:
                #     if k < len(weights):
                #         cur *= weights[k]
            denomi += cur
            if assignments[-1] == 1:
                nomi += cur
        return nomi / denomi

    def knowledge_inference_prob_parallel(self, scores, weights, num_process):
        n = len(scores)
        num_processes = num_process
        processes = []
        output_nomi = multiprocessing.Queue()
        output_denomi = multiprocessing.Queue()

        def compute_factor_value(idx_list, weights, scores, output_nomi, output_denomi):
            n_ = len(scores)
            nomis = []
            denomis = []
            for idx in idx_list:
                assignments = int2assignment(idx, n_)
                cur = 1.0
                for k in range(n_):
                    assi = assignments[k]
                    cur *= (1 - scores[k]) * (1 - assi) + scores[k] * assi
                    for k2 in range(n_):
                        if not (assignments[k] == 1 and assignments[k2] == 0):
                            cur *= weights[k][k2]
                denomis.append(cur)
                if assignments[-1] == 1:
                    nomis.append(cur)
            output_denomi.put(denomis)
            output_nomi.put(nomis)


        start = 0
        batch_size = int(2**n) // num_processes + 1
        for i in range(num_processes):
            process = multiprocessing.Process(target=compute_factor_value, args=(list(range(start, min(start + batch_size, int(2**n)))), weights, scores, output_nomi, output_denomi))
            start += batch_size
            process.start()
            processes.append(process)

        nomi, denomi = 0.0, 0.0
        for i in range(num_processes):
            cur_nomis = output_nomi.get()
            cur_denomis = output_denomi.get()
            nomi += sum(cur_nomis)
            denomi += sum(cur_denomis)

        for process in processes:
            process.join()
        return nomi / denomi

    def scale(self, prob_un):
        # prob_sa = 1. - prob_un
        # prob_un = np.exp(prob_un / self.alpha) / (np.exp(prob_un / self.alpha) + np.exp(prob_sa / self.alpha))
        # if prob_un > 0.4:
        #     prob_un = 1.0
        # else:
        #     prob_un *= 0.5
        # print(prob_un)
        # if prob_un>0.6: #0.55
        #     prob_un=1.0
        return prob_un

    def AC_layer_correction(self, scores_layer, weights_layer):
        prob_unsafe = self.knowledge_inference_prob(scores_layer, weights_layer, parallel=False, num_process=0)
        return prob_unsafe

    def AC_inference(self, scores_graph, weights_graph):
        assert len(scores_graph)==len(weights_graph)
        prob_unsafe = 0.5
        for layer in range(len(scores_graph)):
            scores_graph[layer].append(prob_unsafe)
            prob_unsafe = self.AC_layer_correction(scores_graph[layer], weights_layer=weights_graph[layer])
        return prob_unsafe

    def forward(self, scores_all, parallel_reasoning=False, reasoning_processes=1, AC_inference=False, weight_graph=None):
        probs_unsafe = []
        if not AC_inference:
            for scores, weights in zip(scores_all, self.weights_all):
                prob_unsafe = self.knowledge_inference_prob(scores, weights, parallel_reasoning, reasoning_processes)
                probs_unsafe.append(prob_unsafe)
        else:
            for scores, weights in zip(scores_all, weight_graph):
                probs_unsafe.append(self.AC_inference(scores, weights))
        avg_prob = sum(a * b for a, b in zip(probs_unsafe, self.agg_weights))

        return self.scale(avg_prob)

    def weights_all_to_trainable_weights(self, weights_all):
        tr_weight = []
        for i in range(self.num_models):
            for j in weights_all[i][:-1, -1]:
                tr_weight.append(j)
        for agg_w in self.agg_weights:
            tr_weight.append(agg_w)
        return tr_weight

    def gradient_to_weight_grad_all(self, grad):
        grad_all = []
        for i in range(self.num_models):
            grad_all.append(np.zeros((self.dim_list[i] + 1,self.dim_list[i] + 1)))
        grad_all.append(np.zeros(self.num_models))

        st = 0
        for i in range(self.num_models):
            for j in range(self.dim_list[i]):
                grad_all[i][j, -1] = grad[j+st]
            st += self.dim_list[i]

        for j in range(self.num_models):
            grad_all[self.num_models][j] = grad[st + j]
        return grad_all
    def knowledge_inference_prob_train(self, scores, weights):
        n = len(scores)
        nomi, denomi = 0., 0.
        for idx in range(2 ** n):
            assignments = int2assignment(idx, n)
            cur = 1.0
            for k in range(n):
                assi = assignments[k]
                cur *= (1 - scores[k]) * (1 - assi) + scores[k] * assi
                for k2 in range(n):
                    if not(assignments[k]==1 and assignments[k2]==0):
                        if k2==n-1 and k<n-1:
                            cur *= weights[k]
            denomi += cur
            if assignments[-1] == 1:
                nomi += cur
        return nomi/denomi


    def gradient_to_weights(self, scores_all, label):
        def f(weights_all):
            probs_unsafe = []
            for idx, scores in enumerate(scores_all):
                if idx == 0:
                    weights = weights_all[:self.dim_list[0]]
                elif idx == 1:
                    weights = weights_all[self.dim_list[0]:self.dim_list[0]+self.dim_list[1]]
                else:
                    weights = weights_all[self.dim_list[0]+self.dim_list[1]:-self.num_models]
                prob_unsafe = self.knowledge_inference_prob_train(scores, weights)
                probs_unsafe.append(prob_unsafe)
            avg_prob_unsafe = sum(a * b for a, b in zip(probs_unsafe, weights_all[-self.num_models:]))
            avg_prob_unsafe = self.scale(avg_prob_unsafe)
            loss = -label * gradnp.log(avg_prob_unsafe) - (1-label) * gradnp.log(1-avg_prob_unsafe)

            self.loss_records.append(loss)
            return loss

        # Compute the derivative of the function
        f_prime = grad(f)
        # Evaluate the derivative at a specific point
        weights = self.weights_all_to_trainable_weights(self.weights_all)
        for i in range(len(weights)):
            weights[i] = gradnp.array(weights[i])
        grad_ = f_prime(weights)
        grad_ = self.gradient_to_weight_grad_all(grad_)
        return grad_, self.loss_records[-1]

    def apply_delta_weight(self, delta_weight):
        for i in range(self.num_models):
            self.weights_all[i][:-1, -1] += delta_weight



    def knowledge_inference_prob_train_pseudo(self, scores, weight):
        n = len(scores)
        nomi, denomi = 0., 0.
        for idx in range(2 ** n):
            assignments = int2assignment(idx, n)
            cur = 1.0
            for k in range(n):
                assi = assignments[k]
                cur *= (1 - scores[k]) * (1 - assi) + scores[k] * assi
                for k2 in range(n):
                    if not(assignments[k]==1 and assignments[k2]==0):
                        if k2==n-1 and k<n-1:
                            cur *= weight
            denomi += cur
            if assignments[-1] == 1:
                nomi += cur
        return nomi / denomi

    def pseudo_learning(self, scores_all, label):
        def f(weight):
            probs_unsafe = []
            for idx, scores in enumerate(scores_all):
                prob_unsafe = self.knowledge_inference_prob_train_pseudo(scores, weight)
                probs_unsafe.append(prob_unsafe)
            avg_prob_unsafe = sum(a * b for a, b in zip(probs_unsafe, self.agg_weights))
            avg_prob_unsafe = self.scale(avg_prob_unsafe)
            loss = -label * gradnp.log(avg_prob_unsafe) - (1-label) * gradnp.log(1-avg_prob_unsafe)
            self.loss_records.append(loss)
            return loss

        # Compute the derivative of the function
        f_prime = grad(f)

        # Evaluate the derivative at a specific point
        weight = self.weights_all[0][0][-1]
        weight = gradnp.array(weight)
        grad_ = f_prime(weight)
        return grad_, self.loss_records[-1]