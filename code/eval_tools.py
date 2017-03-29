#!/usr/bin/env python
import numpy as np
import sys
import codecs
import theano
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
np.set_printoptions(precision=2, suppress=True)

__author__ = 'arenduchintala'

if theano.config.floatX == 'float32':
    floatX = np.float32
    intX = np.int32
else:
    floatX = np.float64
    intX = np.int64


__author__ = 'arenduchintala'

def pad_start(_s):
    z = np.zeros((1, _s.shape[1])).astype(floatX)
    sm1 = np.concatenate((z, _s), axis=0)
    sm1 = sm1[:_s.shape[0], :]
    sm1 = floatX(sm1)
    return sm1

def eval_losses(SEQ, seq_model, dh):
    l_per_seq = []
    l_per_user_guess = []
    l_per_user_guess_c = []
    l_per_user_guess_ic = []
    l_per_user_guess_tp = []
    l_per_user_guess_mc = []
    acc_instances = []
    acc_instances_c = []
    acc_instances_ic = []
    acc_instances_mc = []
    acc_instances_tp = []

    _theta_0 = np.zeros((dh.FEAT_SIZE,)).astype(floatX)
    for idx in xrange(len(SEQ)):
        _devX, _devY, _devYT, _devO, _devS = SEQ[idx]
        _devSM1 = pad_start(_devS)
        #seq_model = SimpleLoglinear(dh, reg = options.reg / 100.0, x1=_x1, x2=_x2, adapt = _adapt)
        seq_losses,c_losses, ic_losses, bin_losses = seq_model.get_seq_losses(_devX, _devY, _devYT, _devO, _devS, _devSM1, _theta_0)
        y_hats = seq_model.get_seq_y_hats(_devX, _devY, _devYT, _devO, _devS, _devSM1, _theta_0)
        u_losses = c_losses + ic_losses
        u_losses= u_losses[u_losses > 0.0]
        c_losses = c_losses[c_losses > 0.0]
        ic_losses = ic_losses[ic_losses > 0.0]
        idx_u = np.arange(_devS.shape[0])[_devS[:,(4,5,6)].any(axis=1)] #index of col when 4,5,6 is 1
        tp_idx = np.where(np.logical_and(_devS[:,1] == 1, _devS[:,3] == 0)) 
        mc_idx = np.where(_devS[:,2] == 1) 
        idx_c = np.where(np.logical_or(_devS[:,4] == 1, _devS[:,7] == 1))
        idx_ic = np.where(np.logical_or(_devS[:,5] == 1, _devS[:,8] == 1))
        tp_losses = seq_losses[tp_idx]
        mc_losses = seq_losses[mc_idx]
        sum_u_losses = np.sum(u_losses)
        l_per_seq.append(sum_u_losses)
        l_per_user_guess += u_losses.tolist()
        l_per_user_guess_c += c_losses.tolist()
        l_per_user_guess_ic += ic_losses.tolist()
        l_per_user_guess_tp += tp_losses.tolist()
        l_per_user_guess_mc += mc_losses.tolist()
        y_hat_argmax = np.argmax(y_hats, axis=1)
        y_argmax = np.argmax(_devY, axis=1)
        assert y_argmax.shape[0] == seq_losses.shape[0]
        y_hat_argmax_u = y_hat_argmax[idx_u]
        y_argmax_u = y_argmax[idx_u]
        y_argmax_c = y_argmax[idx_c]
        y_argmax_ic = y_argmax[idx_ic]
        y_argmax_mc = y_argmax[mc_idx]
        y_argmax_tp = y_argmax[tp_idx]
        y_hat_argmax_c = y_hat_argmax[idx_c]
        y_hat_argmax_ic = y_hat_argmax[idx_ic]
        y_hat_argmax_mc = y_hat_argmax[mc_idx]
        y_hat_argmax_tp = y_hat_argmax[tp_idx]
        acc_match_c = [1 if i == j else 0 for i,j in zip(y_argmax_c, y_hat_argmax_c)]
        acc_instances_c += acc_match_c
        acc_match_ic = [1 if i == j else 0 for i,j in zip(y_argmax_ic, y_hat_argmax_ic)]
        acc_instances_ic += acc_match_ic
        acc_match_mc = [1 if i == j else 0 for i,j in zip(y_argmax_mc, y_hat_argmax_mc)]
        acc_instances_mc += acc_match_mc
        acc_match_tp = [1 if i == j else 0 for i,j in zip(y_argmax_tp, y_hat_argmax_tp)]
        acc_instances_tp += acc_match_tp
        acc_match = [1 if i == j else 0 for i,j in zip(y_argmax_u, y_hat_argmax_u)]
        acc_instances += acc_match
    return l_per_seq, l_per_user_guess, l_per_user_guess_c, l_per_user_guess_ic, l_per_user_guess_mc, l_per_user_guess_tp, acc_instances, acc_instances_c, acc_instances_ic, acc_instances_mc, acc_instances_tp


def disp_eval(SEQ, seq_model, dh, trace_file = None, epoch_idx = None, save_model=False):
    ave_total_loss = []
    ave_p_y_u_all = []
    ave_p_y_u = []
    ave_p_y_r = []
    ave_p_y_u_c = []
    ave_p_y_u_ic = []
    ave_p_y_u_ict = []
    acc = []
    acc_instances = []
    _params = seq_model.get_params()
    _max_p = []
    _trace_file = None

    for p in _params:
        _max_p.append("%.3f" % np.max(p) + "," + "%.3f" % np.min(p))
    if trace_file is not None and epoch_idx is not None:
        _trace_file = open(trace_file + '.iter.' + str(epoch_idx), 'w')
    else:
        pass
    _theta_0 = np.zeros((dh.FEAT_SIZE,)).astype(floatX)
    for idx in xrange(len(SEQ)):
        _devX, _devY, _devYT, _devO, _devS = SEQ[idx]
        _devSM1 = pad_start(_devS)
        #seq_model = SimpleLoglinear(dh, reg = options.reg / 100.0, x1=_x1, x2=_x2, adapt = _adapt)
        total_loss, model_loss, all_loss,c_loss,ic_loss,bin_loss = seq_model.get_loss(_devX, _devY, _devYT, _devO, _devS, _devSM1, _theta_0)
        seq_losses,c_losses, ic_losses, bin_losses = seq_model.get_seq_losses(_devX, _devY, _devYT, _devO, _devS, _devSM1, _theta_0)
        y_hats = seq_model.get_seq_y_hats(_devX, _devY, _devYT, _devO, _devS, _devSM1, _theta_0)
        #seq_thetas = seq_model.get_seq_thetas(_devX, _devY, _devYT, _devO, _devS, _devSM1, _theta_0)
        #seq_updates = seq_model.get_seq_updates(_devX, _devY, _devYT, _devO, _devS, _devSM1, _theta_0)
        #seq_g_r = seq_model.get_seq_g_r(_devX, _devY, _devYT, _devO, _devS, _devSM1, _theta_0)
        #seq_g_z = seq_model.get_seq_g_z(_devX, _devY, _devYT, _devO, _devS, _devSM1, _theta_0)
        #y_min_hats = np.min(y_hats, axis=1)
        #y_max_hats = np.max(y_hats, axis=1)
        #y_sum_hats = np.sum(y_hats, axis=1)
        #print y_min_hats
        #print y_max_hats
        #print y_sum_hats
        #print y_hats.shape
        #print seq_thetas.shape
        #pdb.set_trace()
        log_y_hats = np.log(y_hats)
        ll = -np.sum(_devY * log_y_hats, axis=1)
        unrevealed = -(_devS[:,3] - 1)
        ll = unrevealed * ll
        seq_losses = np.reshape(seq_losses, ll.shape)
        p_y_u_all = y_hats[_devY == 1] #probs of all the selections
        p_y_t_all = y_hats[np.arange(y_hats.shape[0]), np.int32(_devYT)] #probs of all the true answers
        assert p_y_t_all.shape == p_y_u_all.shape
        idx_u = np.arange(_devS.shape[0])[_devS[:,(4,5,6)].any(axis=1)] #index of col when 4,5,6 is 1
        idx_u_c = np.arange(_devS.shape[0])[_devS[:,(4,7)].any(axis=1)] #index of col when 4 or 7 is 1 i.e. correct
        idx_u_ic = np.arange(_devS.shape[0])[_devS[:,(5,8)].any(axis=1)] #index of col when 5 is 1 i.e. incorrect
        idx_r = np.arange(_devS.shape[0])[_devS[:,(0,3)].any(axis=1)] #if its an example or any reveal

        p_y_u = p_y_u_all[idx_u] #models prob on all of users answers
        p_y_r = p_y_u_all[idx_r] #models prob on all of users reveals
        p_y_u_c = p_y_u_all[idx_u_c] #models pron on all of users correct answers
        p_y_u_ic = p_y_u_all[idx_u_ic] #models prob on all of users incorrect answers
        p_y_u_ict = p_y_t_all[idx_u_ic] #models prob on all of users incorrect answers
        y_hat_argmax = np.argmax(y_hats, axis=1)
        y_argmax = np.argmax(_devY, axis=1)
        y_hat_argmax_u = y_hat_argmax[idx_u]
        y_argmax_u = y_argmax[idx_u]
        common = [i for i,j in zip(y_argmax_u, y_hat_argmax_u) if i == j]
        acc_match = [1 if i == j else 0 for i,j in zip(y_argmax_u, y_hat_argmax_u)]
        acc_instances += acc_match
        acc.append(len(common))
        if _trace_file is not None:
            _mc = _devS[:,2]
            tp_idx = np.where(np.logical_and(_devS[:,1] == 1, _devS[:,3] == 0)) 
            _tp = np.zeros(_devS.shape[0])
            _tp[tp_idx] = 1 
            _u_correct = _devS[:,4] + _devS[:,7]
            _u_correct = np.reshape(_u_correct, p_y_u_all.shape)
            _u_incorrect = _devS[:,5] + _devS[:,8]
            _u_incorrect = np.reshape(_u_incorrect, p_y_u_all.shape)
            _chance = _devS[:, 9]
            user_plot = np.concatenate((p_y_t_all[:,np.newaxis],
                                        p_y_u_all[:,np.newaxis],
                                        _u_correct[:, np.newaxis],
                                        _u_incorrect[:,np.newaxis],
                                        _mc[:, np.newaxis],
                                        _tp[:,np.newaxis],
                                        _chance[:, np.newaxis],
                                        seq_losses[:, np.newaxis]), axis=1) 
            np.savetxt(_trace_file, user_plot.T, fmt="%.3f") 
        else:
            pass
        ave_p_y_u_all+=p_y_u_all.tolist()
        ave_p_y_u+=p_y_u.tolist()
        ave_p_y_r+=p_y_r.tolist()
        ave_p_y_u_c+=p_y_u_c.tolist()
        ave_p_y_u_ic+=p_y_u_ic.tolist()
        ave_p_y_u_ict+=p_y_u_ict.tolist()
        #ave_total_loss.append(total_loss)
        ave_total_loss.append(model_loss)
    assert len(acc_instances) == len(ave_p_y_u)
    msg = "ave model loss:"  + "%.3f" % np.mean(ave_total_loss) + ",%.3f" % np.std(ave_total_loss) + "," + str(len(ave_total_loss)) + "," + str(len(ave_p_y_u)) +\
        " p_u:" +"%.3f" % np.mean(ave_p_y_u) + ",%.3f" % np.std(ave_p_y_u) + "," + str(len(ave_p_y_u)) + \
        " p_c:" + "%.3f" % np.mean(ave_p_y_u_c)+ ",%.3f" % np.std(ave_p_y_u_c) + "," + str(len(ave_p_y_u_c)) + \
        " p_ic:" + "%.3f" % np.mean(ave_p_y_u_ic)+ ",%.3f" % np.std(ave_p_y_u_ic) + "," + str(len(ave_p_y_u_ic)) + \
        " p_ict:" + "%.3f" % np.mean(ave_p_y_u_ict)+ ",%.3f" % np.std(ave_p_y_u_ict) + "," + str(len(ave_p_y_u_ict)) + \
        " acc:" + "%.3f" % (np.sum(acc) / float(len(ave_p_y_u))) + \
        " params:" + str(len(_max_p))
    #print _params
    #sys.stdout.write(msg +'\n')
    sys.stdout.flush()
    if _trace_file is not None:
        _trace_file.flush()
        _trace_file.close()
    return msg, np.mean(ave_total_loss), np.mean(ave_p_y_u), np.sum(acc) #, acc_instances
