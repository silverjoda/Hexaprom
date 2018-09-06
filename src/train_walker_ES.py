import numpy as np
import gym
import cma
from pid import PDreg
np.random.seed(0)
from weight_distributor import Wdist


# EXP:  CLone experiment and add control heirarchy (first policy controls femurs, femurs control coxas)

def f(w):

    reward = 0
    done = False
    env_obs = env.reset()
    rnn_states = {'h_m' : [0] * n_hidden_m,
                  'h_h1' : [0] * n_hidden_osc,
                  'h_k1' : [0] * n_hidden_osc,
                  'h_f1' : [0] * n_hidden_osc,
                  'h_h2' : [0] * n_hidden_osc,
                  'h_k2' : [0] * n_hidden_osc,
                  'h_f2' : [0] * n_hidden_osc}

    # Observations
    # 0,  1,   2,   3,   4,   5,   6,   7,  8,  9,  10,   11,   12,   13,   14,   15,   16
    # z, th, lj0, lj1, lj2, rj0, rj1, rj2, dx, dz, dth, dlj0, dlj1, dlj2, drj0, drj1, drj2

    pdreg = PDreg(3, 0.05)

    while not done:

        # Master node
        obs = list(env_obs[[0, 1, 8, 9, 10]]) + list(rnn_states['h_h1']) + list(rnn_states['h_h2'])
        h_m = ACT_MULT * np.tanh(np.matmul(wdist.get_w('m_w', w).T, rnn_states['h_m']) + np.matmul(wdist.get_w('m_u', w).T, obs) + wdist.get_w('m_b', w))
        y_m = np.matmul(h_m, wdist.get_w('m_v', w)) + wdist.get_w('m_c', w)

        # h1 node
        obs = list(env_obs[2:3]) + list(y_m[0:n_hidden_osc]) + list(rnn_states['h_k1'])
        h_h1 = ACT_MULT * np.tanh(np.matmul(wdist.get_w('h_w', w).T, rnn_states['h_h1']) + np.matmul(wdist.get_w('h_u', w).T, obs) + wdist.get_w('h_b', w))
        y_h1 = ACT_MULT * np.matmul(wdist.get_w('h_v', w).T, h_h1) + wdist.get_w('h_c', w)

        # h2 node
        obs = list(env_obs[5:6]) + list(y_m[n_hidden_osc:]) + list(rnn_states['h_k2'])
        h_h2 = ACT_MULT * np.tanh(np.matmul(wdist.get_w('h_w', w).T, rnn_states['h_h2']) + np.matmul(wdist.get_w('h_u', w).T,obs) + wdist.get_w('h_b', w))
        y_h2 = np.matmul(wdist.get_w('h_v', w).T, h_h2) + wdist.get_w('h_c', w)

        # k1 node
        obs = list(env_obs[3:4]) + list(rnn_states['h_h1']) + list(rnn_states['h_f1'])
        h_k1 = ACT_MULT * np.tanh(np.matmul(wdist.get_w('k_w', w).T, rnn_states['h_k1']) + np.matmul(wdist.get_w('k_u', w).T, obs) + wdist.get_w('k_b', w))
        y_k1 = np.matmul(wdist.get_w('k_v', w).T, h_k1) + wdist.get_w('k_c', w)

        # k2 node
        obs = list(env_obs[6:7]) + list(rnn_states['h_h2']) + list(rnn_states['h_f2'])
        h_k2 = ACT_MULT * np.tanh(np.matmul(wdist.get_w('k_w', w).T, rnn_states['h_k2']) + np.matmul(wdist.get_w('k_u', w).T,obs) + wdist.get_w('k_b', w))
        y_k2 = np.matmul(wdist.get_w('k_v', w).T, h_k2) + wdist.get_w('k_c', w)

        # f1 node
        obs = list(env_obs[4:5]) + list(rnn_states['h_k1'])
        h_f1 = ACT_MULT * np.tanh(np.matmul(wdist.get_w('f_w', w).T, rnn_states['h_f1']) + np.matmul(wdist.get_w('f_u', w).T, obs) + wdist.get_w('f_b', w))
        y_f1 = np.matmul(wdist.get_w('f_v', w).T, h_f1) + wdist.get_w('f_c', w)

        # f2 node
        obs = list(env_obs[7:8]) + list(rnn_states['h_k2'])
        h_f2 = ACT_MULT * np.tanh(np.matmul(wdist.get_w('f_w', w).T, rnn_states['h_f2']) + np.matmul(wdist.get_w('f_u', w).T, obs) + wdist.get_w('f_b', w))
        y_f2 = np.matmul(wdist.get_w('f_v', w).T, h_f2) + wdist.get_w('f_c', w)

        rnn_states = {'h_m': h_m, 'h_h1': h_h1, 'h_k1': h_k1, 'h_f1': h_f1,
                      'h_h2': h_h2, 'h_k2': h_k2, 'h_f2': h_f2}

        action = [y_h1, y_k1, y_f1, y_h2, y_k2, y_f2]
        action = [a + np.random.randn() * 0.1 for a in action]
        #pd_action = pdreg.update([a[0] for a in action], env_obs[2:8])

        # Step environment
        env_obs, rew, done, _ = env.step(action)

        if animate:
            env.render()

        reward += rew

    return -reward

# Make environment
env = gym.make("Walker2d-v2")
animate = True

# Generate weights
wdist = Wdist()

n_hidden_m = 4
n_hidden_osc = 2

# Master node
wdist.addW((n_hidden_m, n_hidden_m), 'm_w') # Hidden -> Hidden
wdist.addW((5 + 2 * n_hidden_osc, n_hidden_m), 'm_u') # Input -> Hidden
wdist.addW((n_hidden_m, 2 * n_hidden_osc), 'm_v') # Hidden -> Output
wdist.addW((n_hidden_m,), 'm_b') # Hidden bias
wdist.addW((2 * n_hidden_osc,), 'm_c') # Output bias

# Hip node
wdist.addW((n_hidden_osc, n_hidden_osc), 'h_w')
wdist.addW((1 + 2 * n_hidden_osc, n_hidden_osc), 'h_u')
wdist.addW((n_hidden_osc, 1), 'h_v')
wdist.addW((n_hidden_osc,), 'h_b')
wdist.addW((1,), 'h_c')

# Knee node
wdist.addW((n_hidden_osc, n_hidden_osc), 'k_w')
wdist.addW((1 + 2 * n_hidden_osc, n_hidden_osc), 'k_u')
wdist.addW((n_hidden_osc, 1), 'k_v')
wdist.addW((n_hidden_osc,), 'k_b')
wdist.addW((1,), 'k_c')

# Foot node
wdist.addW((n_hidden_osc, n_hidden_osc), 'f_w')
wdist.addW((1 + n_hidden_osc, n_hidden_osc), 'f_u')
wdist.addW((n_hidden_osc, 1), 'f_v')
wdist.addW((n_hidden_osc,), 'f_b')
wdist.addW((1,), 'f_c')

N_weights = wdist.get_N()
print("Nweights: {}, nhm: {}, nho: {}".format(N_weights, n_hidden_m, n_hidden_osc))
W_MULT = 1
ACT_MULT = 5.

w = np.random.randn(N_weights) * W_MULT
#w = [2.307506768153097,-0.8628721159795459,3.3964428566430644,2.9386711164292474,-6.555085257908858,-0.9720738808281748,-2.3808981429570037,2.6033426735432066,4.939753025362945,-4.447314995281907,5.7235592694607185,-0.5070888709360459,6.45142688157377,-2.408773128408832,3.665038817183743,1.3610693455724237,-2.560234404092351,0.8205738614582543,-3.177964024999231,-7.175055534410414,5.246534771138224,4.877602884421626,3.401600054118129,1.9654187269747785,4.057774656459806,-6.197025417360658,-0.2654323915744918,-1.469241625735064,-3.3685428715466728,6.0551638173059725,5.448451430463363,6.4021573257047075,-4.4846173760138095,-7.432344561767951,-9.261366794017542,-4.256746391508243,-4.853955042496246,0.530465915324472,-0.4834847538160607,0.46395606701044795,-6.331488258219134,-5.037121576095786,6.129667349867583,0.3770466294209443,4.422055105511893,4.8700994152522545,-0.6657231649932437,-4.18372104699389,-4.227593328890637,-3.953033135354146,-3.74078686439446,-2.3797538928088793,-0.5026462044803268,3.8889627295647906,-7.819819433253633,0.41107121995679585,2.3150553405478074,-0.11194145578623937,3.597662753630561,-6.793173723878911,-6.517679956484592,1.410650153057905,-3.0718714326493273,-1.267236580165312,0.2524848906590489,-2.0050129644405636,1.776029748366181,4.460161617320464,-1.256027716273427,-3.4272617445867612,-1.7041074644058412,-3.1296992907499517,-4.684618215978085,-0.3197027798237339,-9.407171894132622,1.1432545590914267,2.7760385140950854,3.2131649734515446,1.7304242430454506,2.6126768445093917,-4.239318555236354,-13.282063180858385,6.24467509380327,-1.5453007348569756,-1.1013277210989894,3.5148740512914345,3.6817805061413322,1.9353913254397552,-1.216950739651666,1.552002598899911,0.3301972159213128,1.3898746058019367,1.852876250221078,-2.7067900946650285,0.028209596039986917,-1.6258599482261453,-0.3212852659648154,1.5134865074201294,-5.8965842628984895,2.143189740688841,6.356564776135079,0.82973927463572,6.37527535918108,3.0522082990672517,-10.7306383653743,-1.146479691837745,-6.6170899298953705,1.9486141705543711,4.380873556469335,1.4645755866531533,-0.19254103215892895,2.376046066633171,1.029194094057743,-0.3779001925949498,1.5470700462491382,4.946385254321469,4.885499432228888,7.848577346016665,1.871054251403399,1.5151298999789082,-6.6984481200440165,1.617315294575195,-3.9357057989127693,-1.2294574409117784,0.08076785965673958,-0.5311616257168297,-1.6287498239666733,2.0485317445169584,0.7775631725046799]
#w=np.array(w)

es = cma.CMAEvolutionStrategy(w, 0.5)
try:
    es.optimize(f, iterations=10000)
except KeyboardInterrupt:
    print("User interrupted process.")
es.result_pretty()


print('w = [' + ','.join(map(str, es.result.xbest)) + '] '  + str(es.result.fbest), file=open("walker_weights.txt", "a"))
