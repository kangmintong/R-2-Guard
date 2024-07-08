from data_loading import fuse_advbench_hotpot



if __name__=="__main__":
    fuse_advbench_hotpot(adv_path='./data/advbench/harmful_strings.csv', output_path='./data/advbench/advbench_string_hotpot.jsonl', benign_num=1000)
    fuse_advbench_hotpot(adv_path='./data/advbench/harmful_behaviors.csv', output_path='./data/advbench/advbench_behaviour_hotpot.jsonl', benign_num=1000)