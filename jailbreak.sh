#python jailbreak.py --attack_method tap --attack_model llamaguard --inference_model mistral --dataset advbench_string --data_size -1 --gpu_attack_model cuda:0 --gpu_inference_model cuda:1
#python jailbreak.py --attack_method tap --attack_model openai_mod --inference_model mistral --dataset advbench_string --data_size -1 --gpu_attack_model cuda:6 --gpu_inference_model cuda:7
#python jailbreak.py --attack_method tap --attack_model toxicchat-T5 --inference_model mistral --dataset advbench_string --data_size -1 --gpu_attack_model cuda:4 --gpu_inference_model cuda:5
#python jailbreak.py --attack_method tap --attack_model r2guard --inference_model mistral --dataset advbench_string --data_size -1 --gpu_attack_model cuda:1 --gpu_inference_model cuda:3

#python jailbreak.py --attack_method pair --attack_model llamaguard --inference_model mistral --dataset advbench_string --data_size -1 --gpu_attack_model cuda:0 --gpu_inference_model cuda:1
#python jailbreak.py --attack_method pair --attack_model openai_mod --inference_model mistral --dataset advbench_string --data_size -1 --gpu_attack_model cuda:0 --gpu_inference_model cuda:2
#python jailbreak.py --attack_method pair --attack_model toxicchat-T5 --inference_model mistral --dataset advbench_string --data_size -1 --gpu_attack_model cuda:3 --gpu_inference_model cuda:3
#python jailbreak.py --attack_method pair --attack_model ensemble --inference_model mistral --dataset advbench_string --data_size -1 --gpu_attack_model cuda:3 --gpu_inference_model cuda:4
#python jailbreak.py --attack_method pair --attack_model r2guard --inference_model mistral --dataset advbench_string --data_size -1 --gpu_attack_model cuda:4 --gpu_inference_model cuda:5

#python jailbreak.py --attack_method autodan --attack_model toxicchat-T5 --inference_model mistral --dataset advbench_behaviour --data_size -1 --gpu_attack_model cuda:1 --gpu_inference_model cuda:1
#python jailbreak.py --attack_method autodan --attack_model llamaguard --inference_model mistral --dataset advbench_behaviour --data_size -1 --gpu_attack_model cuda:2 --gpu_inference_model cuda:3
#python jailbreak.py --attack_method autodan --attack_model openai_mod --inference_model mistral --dataset advbench_behaviour --data_size -1 --gpu_attack_model cuda:7 --gpu_inference_model cuda:7
#python jailbreak.py --attack_method autodan --attack_model r2guard --inference_model mistral --dataset advbench_behaviour --data_size -1 --gpu_attack_model cuda:1 --gpu_inference_model cuda:7

