#!/usr/bin/python
"""
run cmdStan over models and data
collect statistics
"""

import os
import os.path
import sys
import subprocess
import time
import shutil
import gzip
import re
import random
import shlex

def stop_err( msg ):
    sys.stderr.write( '%s\n' % msg )
    sys.stderr.write( 'exit now ( %s)\n' % time.strftime('%x %X %Z'))
    sys.exit()

def run_JAGS():
    print("run JAGS")
    
def run_Stan_NUTS( model, datafile, num_warmup, num_samples, num_runs, job_num ):
    # setup stan command
    command = model + " sample num_warmup=" + str(num_warmup) + " num_samples=" + str(num_samples) + " data file=" + datafile
    # don't save warmup, number of warmups, samples
    print(command)
    run_Stan(command, datafile, "nuts", num_runs, job_num)
        
def run_Stan_Stretch_Ensemble( model, datafile, num_warmup, num_samples, num_runs, job_num ):
    # setup stan command
    command = model + " sample algorithm=stretch_ensemble num_warmup=" + str(num_warmup) + " num_samples=" + str(num_samples) + " data file=" + datafile
    # don't save warmup, number of warmups, samples
    print(command)
    run_Stan(command, datafile, "stretch_ensemble", num_runs, job_num)
            
def run_Stan_Walk_Ensemble( model, datafile, num_warmup, num_samples, num_runs, job_num ):
    # setup stan command
    command = model + " sample algorithm=walk_ensemble num_warmup=" + str(num_warmup) + " num_samples=" + str(num_samples) + " data file=" + datafile
    # don't save warmup, number of warmups, samples
    print(command)
    run_Stan(command, datafile, "walk_ensemble", num_runs, job_num)
    
# run_Stan:  does N runs of cmdStan
# collects run times and parameter stats from each run
# optionally saves sampler output
def run_Stan ( stan_cmd, datafile, method, num_runs, job_num) :
    saveOutputCsv = False
    datapath = datafile.split('/')
    datafilename = datapath[len(datapath)-1]
    datafilesplit = datafilename.split('.')
    modelname = datafilesplit[0]
    params_filename = "results/stats_param_" + modelname + "_" + method + ".txt"
    params_fh = open(params_filename,'w')
    times_filename = "results/stats_time_" + modelname + "_" + method + ".txt"
    times_fh = open(times_filename,'w')
    rams_filename = "results/stats_ram_" + modelname + "_" + method + ".txt"
    rams_fh = open(rams_filename,'w')

    for i in range(num_runs):
        runname = modelname + "_" + method + "_" + str(i+1)
        print("\n" + runname)

        # FIXME: on mac, use /usr/bin/time -l 
        stan_cmd2 = "/usr/bin/time -v " + stan_cmd  
        binprint_cmd = "bin/print"

        chains = list()
        chains_out = list()
        chains_err = list()
        seed = random.randint(1,10000)
        for j in range(4):
            outputfile = "output" + str(j+4*job_num) + ".csv"
            binprint_cmd = ' '.join([binprint_cmd,outputfile])
            stan_cmd3 = stan_cmd2 + " random seed=" + str(seed) + " id=" + str(j+1) + " output file=" + outputfile
            args = shlex.split(stan_cmd3)
            print('start chain %d ' % j + stan_cmd3 + ' ( %s)' % time.strftime('%x %X %Z'))
            chains.append(subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE))
            chains_out.append("")
            chains_err.append("")
            
        for j in range(len(chains)):
            chains_out[j], chains_err[j] = chains[j].communicate()
            retcode = chains[j].returncode
            if (retcode != 0):
                msg = 'chain %d ' % j + stan_cmd + ' failed'
                stop_err(msg)
            print('finish chain %d ' % j + stan_cmd + ' ( %s)' % time.strftime('%x %X %Z'))
            # get max RAM from stdout
            for line in chains_err[j].decode().split('\n'):
                # for mac users use
                # if re.search("maximum resident set size",line):
                if re.search("Maximum resident set size",line):
                    tokens = line.split()
                    # for mac users use
                    # rams_fh.write(tokens[0])
                    ram_fh.write(line)
                    ram_fh.write('\n')
                    rams_fh.write(tokens[5])
                    rams_fh.write(' ')
                    break;
            # grep output.csv for time info
            outputfile = "output" + str(j+4*job_num) + ".csv"
            for line in open(outputfile,'r'):
                if re.search("(Warm-up)",line):
                    tokens = line.split()
                    times_fh.write(" ")
                    times_fh.write(tokens[3])
                    times_fh.write(" ")
                    break
            for line in open(outputfile,'r'):
                if re.search("(Sampling)",line):
                    tokens = line.split()
                    times_fh.write(" ")
                    times_fh.write(tokens[1])
                    times_fh.write(" ")
                    break
        rams_fh.write('\n')
        times_fh.write('\n')
                
            
        # run bin/print on output.csv
        p1 = subprocess.Popen(binprint_cmd.split(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        stdout, stderr = p1.communicate()
        retcode = p1.returncode
        if (retcode != 0):
            msg = binprint_cmd + ' failed'
            stop_err(msg)
        munge_binprint(stdout,params_fh)

        if saveOutputCsv:
            gzname = runname + "_output.csv.gz"
            with open('output.csv', 'rb') as f_in:
                with gzip.open(gzname, 'wb') as f_out:
                    f_out.writelines(f_in)
        
        
    params_fh.close()
    rams_fh.close()
    times_fh.close()

# munge_binprint:  scrapes model param stats from bin/print output
def munge_binprint( output, fh ):
    header = "Mean"
    sampler_stats = "lp__,accept_stat__,stepsize__,treedepth__,n_leapfrog__,n_divergent__,scale__".split(',')
    skip = True
    lines = output.decode().split('\n')
    
    for line in lines:
        tokens = line.split()
        if skip:
            if len(tokens) > 0 and tokens[0] in header:
                skip = False
        else:
            if len(tokens) > 0 and not (tokens[0] in sampler_stats):
                fh.write(line)
                fh.write('\n')
            elif len(tokens) == 0:
                skip = True

def main():
    if (len(sys.argv) < 8):
        stop_err("usage: samplerEval <model> <datafile> <method> <num_warmup> <num_samples> <num_runs> ")

    model = sys.argv[1]
    datafile = sys.argv[2]
    method = sys.argv[3]
    num_warmup = int(sys.argv[4])
    num_samples = int(sys.argv[5])
    num_runs = int(sys.argv[6])
    job_num = int(sys.argv[7])

    if method.lower().startswith("nuts"):
        run_Stan_NUTS(model, datafile, num_warmup, num_samples, num_runs, job_num)
    elif method.lower().startswith("str"):
        run_Stan_Stretch_Ensemble(model, datafile, num_warmup, num_samples, num_runs, job_num)
    elif method.lower().startswith("walk"):
        run_Stan_Walk_Ensemble(model, datafile, num_warmup, num_samples, num_runs, job_num)
    # run model using JAGS - 
    
    print('\nall processing completed ( %s)\n' % time.strftime('%x %X %Z'))

if __name__ == "__main__":
    main()
    
