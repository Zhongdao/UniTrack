import os
import socket
import time

def test(model, L=20, K=10, T=0.07, R=12, opts=[], gpu=0, force=False, dryrun=False):
    R = int(R)

    if os.path.exists(model):
        if os.path.isdir(model):
            model = sorted([f"{model}/{s}" for s in os.listdir(model) if 'model_' in s], key=os.path.getmtime)[::-1][0]

        if 'rel_left_drop' in model or 'zero' in model:
            model_type = 'scratch_zeropad'
        else:
            model_type = 'scratch'

        model_str = f"--model-type {model_type} --resume {model}"
        model_name = '_'.join(model.split('/')[1:]) #.replace('/', '_')
    else:
        model_str = '--model-type %s' % model
        model_name = model

    outdir = 'tmp/'   
    davis2017path = '/home/wangzd/datasets/uvc/DAVIS/'
    datapath = davis2017path 
    
    model_name = "%s_L%s_K%s_T%s_R%s_opts%s_M%s" % \
                    (str(int(time.time()))[-4:], L, K, T, R, ''.join(opts), model_name) 
    time.sleep(1)

    opts = ' '.join(opts)
    cmd = ""
    
    outfile = f"{outdir}/converted_{model_name}/global_results-val.csv"
    online_str = '_online' if '--finetune' in opts else ''

    if ((not os.path.isfile(outfile)) or force):
        print('Testing', model_name)
        if (not os.path.isdir(f"{outdir}/results_{model_name}")) or force:
            cmd += f" python test.py --filelist eval/davis_vallist.txt {model_str} \
                    --topk {K} --radius {R}  --videoLen {L} --temperature {T} --save-path {outdir}/results_{model_name} \
                    --workers 5  {opts} --gpu-id {gpu} && "

        convert_str = f"python eval/convert_davis.py --in_folder {outdir}/results_{model_name}/ \
                --out_folder {outdir}/converted_{model_name}/ --dataset {datapath}"

        eval_str = f"python {davis2017path}/evaluation_method.py --task semi-supervised \
                --results_path  {outdir}/converted_{model_name}/ --set val --davis_path {datapath}"
        
        cmd += f" {convert_str} && {eval_str}"
        print(cmd)

        if not dryrun:
            os.system(cmd)

def run(models, L, K, T, R, size, finetune, slurm=False, force=False, gpu=-1, dryrun=False):
    import itertools

    base_opts = ['--cropSize', str(size),]

    if finetune > 0:
        base_opts += ['--head-depth', str(0), '--use-res4', '--finetune', str(finetune)]
    else:
        base_opts += ['--head-depth', str(-1)]

    opts = [base_opts]
    prod = list(itertools.product(models, L, K, T, R, opts))
    
    if slurm:
        for p in prod:
            cmd = f"sbatch --export=model_path={p[0]},L={p[1]},K={p[2]},T={p[3]},R={p[4]},size={size},finetune={finetune} /home/ajabri/slurm/davis_test.sh"
            print(cmd)
            os.system(cmd)
    else:
        print(prod)
        for i in range(0, len(prod)):
            test(*prod[i], 0 if gpu == -1 else gpu, force, dryrun=dryrun)
                        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', default=[], type=str, nargs='+',
                        help='(list of) paths of models to evaluate')

    parser.add_argument('--slurm', default=False, action='store_true')
    parser.add_argument('--force', default=False, action='store_true')
    parser.add_argument('--dryrun', default=False, action='store_true')

    parser.add_argument('--L', default=[20], type=int, nargs='+')
    parser.add_argument('--K', default=[10], type=int, nargs='+')
    parser.add_argument('--T', default=[0.07], type=float, nargs='+')
    parser.add_argument('--R', default=[12], type=float, nargs='+')
    parser.add_argument('--cropSize', default=-1, type=int)

    parser.add_argument('--finetune', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    
    args = parser.parse_args()
    
    if len(args.model_path) == 0:
        args.model_path = models

    if args.slurm:
        run(args.model_path, args.L, args.K, args.T, args.R, args.cropSize, args.finetune,
            slurm=args.slurm, force=args.force, dryrun=args.dryrun)
    else:
        run(args.model_path, args.L, args.K, args.T, args.R, args.cropSize, args.finetune,
            force=args.force, gpu=args.gpu, dryrun=args.dryrun)
