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

def disp_eval(SEQ, seq_model, dh, trace_file = None, epoch_idx = None):
    ave_total_loss = []
    ave_p_y_u_all = []
    ave_p_y_u = []
    ave_p_y_r = []
    ave_p_y_u_c = []
    ave_p_y_u_ic = []
    ave_p_y_u_ict = []
    #user_traces = []
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
        #seq_model = SimpleLoglinear(dh, reg = options.reg / 100.0, x1=_x1, x2=_x2, adapt = _adapt)
        total_loss,all_loss,r_loss,c_loss,ic_loss = seq_model.get_loss(_devX, _devY, _devO, _devS, _theta_0)
        seq_losses,r_losses, c_losses, ic_losses = seq_model.get_seq_losses(_devX, _devY, _devO, _devS, _theta_0)
        y_hats = seq_model.get_seq_y_hats(_devX, _devY, _devO, _devS, _theta_0)
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
        if _trace_file is not None:
            _u_correct = _devS[:,4] + _devS[:,7]
            _u_correct = np.reshape(_u_correct, p_y_u_all.shape)
            _u_incorrect = _devS[:,5] + _devS[:,8]
            _u_incorrect = np.reshape(_u_incorrect, p_y_u_all.shape)
            user_plot = np.concatenate((p_y_t_all[:,np.newaxis],p_y_u_all[:,np.newaxis], _u_correct[:, np.newaxis], _u_incorrect[:,np.newaxis]), axis=1) 
            np.savetxt(_trace_file, user_plot.T, fmt="%.3f") 
        else:
            pass
        #summary  = np.concatenate((ll[:,np.newaxis], seq_losses[:,np.newaxis], r_losses[:,np.newaxis], c_losses[:,np.newaxis], ic_losses[:,np.newaxis], p_y_u_all[:,np.newaxis], _devS[:,[3,4,5,6,7,8]]), axis=1)
        #print summary
        #print p_y_u.mean(), p_y_u_c.mean(), p_y_u_ic.mean(), 
        #pdb.set_trace()
        ave_p_y_u_all+=p_y_u_all.tolist()
        ave_p_y_u+=p_y_u.tolist()
        ave_p_y_r+=p_y_r.tolist()
        ave_p_y_u_c+=p_y_u_c.tolist()
        ave_p_y_u_ic+=p_y_u_ic.tolist()
        ave_p_y_u_ict+=p_y_u_ict.tolist()
        #user_plot = np.concatenate((idx_u[:,np.newaxis], p_y_u[:, np.newaxis], f_u[:,np.newaxis]), axis=1)
        #user_traces.append(user_plot)
        ave_total_loss.append(total_loss)
    msg = "ave total loss:"  + "%.3f" % np.mean(ave_total_loss) +\
        " p_u:" +"%.3f" % np.mean(ave_p_y_u) + ",%.3f" % np.std(ave_p_y_u) + \
        " p_c:" + "%.3f" % np.mean(ave_p_y_u_c)+ ",%.3f" % np.std(ave_p_y_u_c) + \
        " p_ic:" + "%.3f" % np.mean(ave_p_y_u_ic)+ ",%.3f" % np.std(ave_p_y_u_ic) + \
        " p_ict:" + "%.3f" % np.mean(ave_p_y_u_ict)+ ",%.3f" % np.std(ave_p_y_u_ict) + \
        " params:" + '--'.join(_max_p)
    sys.stdout.write(msg +'\n')
    sys.stdout.flush()
    if _trace_file is not None:
        _trace_file.flush()
        _trace_file.close()
    return True
