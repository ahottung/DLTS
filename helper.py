import os
import os.path
import numpy as np
import logging
import random
import glob

def parse_file_pointer(fp, dummy_tiers):
    lines = [ll.strip() for ll in fp]
    ii = 0
    tiers = 0
    stacks = 0
    runtime = 0
    labels = []
    res = []
    while ii < len(lines):
        line = lines[ii]
        if not line.startswith("AA"):
            ii += 1
            continue
        line = line[3:]
        if tiers == 0:
            sp = line.split()
            tiers = int(sp[1])
            stacks = int(sp[2])
            if (len(sp) > 3):
                runtime = float(sp[3])
            ii += tiers + 1
        else:
            if "from" not in line:
                print("Error parsing input")
                ii += len(lines)
            else:
                sp = line.split()[-1].split("/")
                sfrom = int(sp[-2])
                sto = int(sp[-1])
                label = [0] * (stacks * (stacks-1))
                index = sfrom*(stacks-1) + sto
                if sto > sfrom:
                    index -= 1
                label[index] = 1
                labels.append(label)
                cells = []
                for tt in range(tiers+1):
                    cells.extend([int(cc) for cc in lines[ii - tt][2::].split(",")[:-1]])
                cells = np.reshape(cells,((tiers, -1)))

                #create dummy tiers
                cc = list(range(cells.max()+1,cells.max()+dummy_tiers*stacks+1))
                for d in range(dummy_tiers):
                    cells = np.r_[[cc[0:stacks]], cells]
                    cc = cc[stacks:]

                cells = np.transpose(cells)
                cells = np.reshape(cells,-1)
                cells = np.round(cells * ((stacks*(tiers-2+dummy_tiers))/np.max(cells))).astype(int)
#                 #cells = cells / np.max(cells)
                res.append(cells)
                ii += tiers + 1
    labels_v = list(range(len(labels),0,-1))
    return (stacks, tiers+dummy_tiers, res, labels, labels_v, runtime)

def parse_dir(ddir, max_n, seed, dummy_tiers):
    logging.info("Start reading instances at: "+str(ddir))
    res = []
    labels = []
    labels_v = []
    ref_rt = []
    stacks = tiers = None
    i = 0
    random.seed(seed)
    files = sorted([os.path.basename(ii) for ii in glob.glob("{0}/*.out".format(ddir))])
    random.shuffle(files)
    random.seed()
    for ff in files:
        with open(os.path.join(ddir,ff), 'r') as fp:
            stacks,tiers,rr,ll,ll_v,rt = parse_file_pointer(fp, dummy_tiers)
            res.extend(rr)
            labels.extend(ll)
            labels_v.extend(ll_v)
            ref_rt.append(rt)
            i += 1
        if i >= max_n:
            break
    logging.info("Finished reading instances")
    return stacks, tiers, res, labels, labels_v, ref_rt
