import numpy as np

# each speaker is modeled via a single GMM

def get_llks(X_mfcc, array_of_Cs, array_of_gaussian_means, array_of_gaussian_covs):
    """
        X_mfcc = matrix of shape TxD

        array_of_Cs: shape NxGx1 
        array_of_gaussian_means: shape NxGxD
        array_of_gaussian_covs: shape NxGxDxD

        G = number of gaussians in speaker GMM
        N = number of speaker GMMs

        returns: 
            p(X_mfcc|theta_i) for the GMM
    """

    # P(x_k | theta ) = summation_i[  c_i * p_i( x_k | mean_i, cov_i) ]
    # p(X | theta) = product_k[ P(x_k | theta) ]
    # log p(X | theta) = summation_k[ log(P(x_k | theta)) ] 

    covs = np.expand_dims(array_of_gaussian_covs, axis=2)
    means = np.expand_dims(array_of_gaussian_means, axis=2)
    
    Z = X_mfcc - means
    exponent_terms = -0.5*np.multiply(np.matmul(Z, np.linalg.inv(covs).squeeze()), Z).sum(axis=-1)
    cov_dets = np.linalg.det(covs)

    pi_x = np.exp(exponent_terms)/pow(2*np.pi, -covs.shape[-1]/2)*np.sqrt(cov_dets) 
    p_x = (array_of_Cs*pi_x).sum(axis=1)
    log_P_X = np.log(p_x).sum(axis=1)

    return log_P_X


def calculate_probs(x_mfcc, Cs, means, covs):
    """
        x_mfcc: an vector of shape 1xD or 2D matrix of shape TxD

        Cs: shape NxGx1 
        means: shape NxGxD
        covs: shape NxGxDxD

        returns: P(z = i| x, theta) shape NxT for each GMM i 

        P(z = i| X, theta) = P_i(X| theta_i)/summation_i[P_i(x| theta_i)]
        
    """
    covs = np.expand_dims(covs, axis=2)
    means = np.expand_dims(means, axis=2)
    
    z = x_mfcc - means
    exponent_terms = -0.5*np.multiply(np.matmul(z, np.linalg.inv(covs).squeeze()), z).sum(axis=-1)
    cov_dets = np.linalg.det(covs)

    pi_x = np.exp(exponent_terms)/pow(2*np.pi, -covs.shape[-1]/2)*np.sqrt(cov_dets)
    p_x = (Cs * pi_x).sum(axis=1)
    P_X = np.exp(np.log(p_x).sum(axis=1, keepdims=True))
    probs = P_X/np.sum(P_X, keepdims=True)

    return probs


def eta(t, a=0.999, b=1000):
    return 1/(a*t + b)


def online_adaptation(X_mfcc, single_spk_gmm_params, globaltime, anb = (0.999, 1000)):
    """ adapts the Gmms 
        X_mfcc: a 2D matrix of shape TxD
        single_spk_gmm_params: tupple or list (array_of_Cs, array_of_gaussian_means, array_of_gaussian_covs)
        globaltime: current time step
        returns: updated Gmm params in a tupple, current globaltime

        <<f>>(t) = <<f>>(t-1) + eta(t)*[f(t)Pi(t) - <<f>>(t-1)]
    """

    LT = 300
    local_time = 0

    Cs = single_spk_gmm_params[0][0]
    means = single_spk_gmm_params[1][0]
    covs = single_spk_gmm_params[2][0]
    
    stat_1s = Cs
    stat_x = means*Cs
    stat_x2 = (covs + np.matmul(means[:,:,np.newaxis], means[:, np.newaxis, :]))*Cs[:, np.newaxis, :]
    # stat_x2 = stat_x2 * np.concatenate([[np.identity(stat_x2.shape[1])]]*stat_x2.shape[0])

    for x_mfcc in X_mfcc:
        # calculate <<1>>(t)
        # calculate <<x>>(t)
        # calculate <<x^2>>(t)
        # update Cs, means, covs

        globaltime = globaltime+1

        if local_time > LT: continue

        eta_t = eta(local_time, anb[0], anb[1])

        pi = calculate_probs(x_mfcc, Cs[:, None], means[:, None], covs[:, None])
        stat_1s = stat_1s + eta_t*(pi - stat_1s)
        stat_x = stat_x + eta_t*(x_mfcc*pi - stat_x)

        # sph = np.diag(np.diag(np.matmul(x_mfcc[:, None], x_mfcc[None,:])))
        sph = np.matmul(x_mfcc[:, None], x_mfcc[None,:]) * pi[:, None, :]
        stat_x2 = stat_x2 + eta_t*(sph - stat_x2)        

        Cs = stat_1s
        means = stat_x/stat_1s
        covs = stat_x2/stat_1s[:,:, np.newaxis] - np.matmul(means[:,:,np.newaxis], means[:, np.newaxis, :])

        local_time += 1

    return [Cs[None, :], means[None, :], covs[None, :]], globaltime


def novelity_detection(X_mfcc, threshold, spk_gmm_params, gen_gmm_params):
    """
        X_mfcc: a 2D matrix of shape TxD
        threshold: threshold for deciding novelity

        spk_gmm_params and gen_gmm_params denote a set of GMMs
        spk_gmm_params: tupple or list (array_of_Cs, array_of_gaussian_means, array_of_gaussian_covs)
        gen_gmm_params: tupple or list (gender_Cs, gender_means, gender_covs)

        array_of_Cs / gender_Cs: shape NxGx1 
        array_of_gaussian_means / gender_means: shape NxGxD
        array_of_gaussian_covs / gender_covs: shape NxGxDxD

        time: current inference time

        returns: 
            if a new speaker is detected:
                new spk_gmm_params, num of speakers + 1
            else:
                detected_spk_gmm_params, detected_spk_gmm_index
    """

    gen_Cs, gen_means, gen_covs = gen_gmm_params
    gender_llks = get_llks(X_mfcc, gen_Cs, gen_means, gen_covs)

    male_or_female = np.argmax(gender_llks) # 0 -> male

    if len(spk_gmm_params[0]) == 0: # first speaker
        return [ np.array([gen_Cs[male_or_female]]), np.array([gen_means[male_or_female]]), np.array([gen_covs[male_or_female]]) ], 0, male_or_female, -1, -1

    spk_Cs, spk_means, spk_covs = spk_gmm_params
    speaker_llks = get_llks(X_mfcc, spk_Cs, spk_means, spk_covs)

    index_gaussian_maxPsp = np.argmax(speaker_llks)
    log_of_P_sp = speaker_llks[index_gaussian_maxPsp]
    log_of_P_gen = gender_llks[male_or_female]

    # if len(speaker_llks) > 1:
    #     log_pavg = np.log((np.exp(speaker_llks).sum() - np.exp(log_of_P_sp))/(len(speaker_llks)-1) + 1e-200)
    #     print(np.exp(log_of_P_sp))
    # else:
    #     log_pavg = log_of_P_sp

    log_of_Liklihood_ratio = log_of_P_sp - log_of_P_gen# - log_pavg

    if log_of_Liklihood_ratio < threshold:
        # enroll new speaker
        return [ 
            np.array([gen_Cs[male_or_female]]), 
            np.array([gen_means[male_or_female]]), 
            np.array([gen_covs[male_or_female]]) 
            ], len(speaker_llks), male_or_female, index_gaussian_maxPsp, log_of_Liklihood_ratio
    else:
        return [ 
            spk_Cs[index_gaussian_maxPsp][None, :], 
            spk_means[index_gaussian_maxPsp][None, :], 
            spk_covs[index_gaussian_maxPsp][None, :] 
            ], index_gaussian_maxPsp, male_or_female, index_gaussian_maxPsp, log_of_Liklihood_ratio


def update_params(adapted_gmm, gmm_index, spk_gmm_params, label_for_gmm, last_seen_spk, current_time):
    """
        modifies spk_gmm_params, label_for_gmm and last_seen_spk to enroll a new speaker or update an existing speaker
    """
    if gmm_index == len(spk_gmm_params[0]): # new speaker
        last_seen_spk.append(0)
        if (gmm_index == 0):
            label_for_gmm.append(0)
            for i in range(3):
                spk_gmm_params[i] = adapted_gmm[i]
        else:
            label_for_gmm.append(label_for_gmm[-1]+1)
            for i in range(3):
                spk_gmm_params[i] = np.concatenate([spk_gmm_params[i], adapted_gmm[i]])
    
    else:
        for i in range(3):
            spk_gmm_params[i][gmm_index] = adapted_gmm[i]

    # gmm_index is in range now
    last_seen_spk[gmm_index] = current_time


def remove_dormant_speakers(spk_gmm_params, label_for_gmm, last_seen_spk, current_time, threshold = 100000):
    """
        removes those speakers which have not been seen since time - threshold mfcc vectors
    """
    i = 0
    while i < len(last_seen_spk):
        if current_time - last_seen_spk[i] > threshold:
            for k in range(3): 
                spk_gmm_params[k] = np.delete(spk_gmm_params[k], i, 0)
            label_for_gmm.pop(i)
            last_seen_spk.pop(i)
        else:
            i+=1




# demo run

import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture


X_male, _ = make_blobs(n_samples=100, centers=[[0,0]], cluster_std=[5], n_features=2, random_state=0)
X_female, _ = make_blobs(n_samples=100, centers=[[15, 15]], cluster_std=[5], n_features=2)

plt.scatter(X_male[:,0], X_male[:,1], marker='o')
plt.scatter(X_female[:,0], X_female[:,1], marker='+')
plt.show()

male_gm = GaussianMixture(n_components=3, random_state=0).fit(X_male)
female_gm = GaussianMixture(n_components=3, random_state=0).fit(X_female)

gender_gmm_params = (
    np.array([male_gm.weights_, female_gm.weights_])[:, :, None], 
    np.concatenate([[male_gm.means_], [female_gm.means_]]), 
    np.concatenate([[male_gm.covariances_], [female_gm.covariances_]]), 
    )
print("gender gmm params\n", gender_gmm_params)

spk_centers = [[-1,-1], [25,15], [10,10]]
spk_stds = [3, 3, 3]
spk_nsamps = [250, 300, 200]
spk_sets = []
for spk_nsamp, spk_center, spk_std in zip(spk_nsamps, spk_centers, spk_stds):
    X_speaker, speaker = make_blobs(n_samples=spk_nsamp, centers=[spk_center], cluster_std=[spk_std], n_features=2)
    spk_sets.append(X_speaker)
    plt.scatter(X_speaker[:,0], X_speaker[:,1], marker=len(spk_sets))
plt.show()

segments = [spk_sets[0][:100], spk_sets[1][0:50], spk_sets[2][0:150], spk_sets[0][100:], spk_sets[1][5:100], spk_sets[2][150:], spk_sets[1][100:]]
labels = [0, 1, 2, 0, 1, 2, 1]

threshold = 2
time=0
est_labels = []

spk_gmm_params = [ [],[],[] ]
label_for_gmm = []
last_seen_spk = []
for feature_seg in segments:

    # novelity
    selected_gmm, gmm_index = novelity_detection(feature_seg, threshold, spk_gmm_params, gender_gmm_params)

    # continual learning
    adapted_gmm, time = online_adaptation(feature_seg, selected_gmm, time)

    # enroll if new speaker
    update_params(adapted_gmm, gmm_index, spk_gmm_params, label_for_gmm, last_seen_spk)

    # append estimate
    est_labels.append(label_for_gmm[gmm_index])

    # kickout a speaker if dormant
    remove_dormant_speakers(spk_gmm_params, label_for_gmm, last_seen_spk, time)

print(est_labels)




















# for dummy runs
import numpy as np

array_of_gaussian_means = np.array([

    [1,1],
    [2,1],
    [1,2.5],
    [2,2],

])

array_of_gaussian_means1 = np.array([

    [3,3],
    [3.5,3.5],
    [3,2.5],
    [3,3.5],

])

array_of_gaussian_covs = np.array([

    [ [1, 0],[0,1] ],
    [ [1, 0],[0,1] ],
    [ [1, 0],[0,1] ],
    [ [1, 0],[0,1] ],

])

array_of_gaussian_covs1 = np.array([

    [ [1, 0],[0,1] ],
    [ [1, 0],[0,1] ],
    [ [1, 0],[0,1] ],
    [ [1, 0],[0,1] ],

])

array_of_Cs = np.array([
    [0.25], [0.25], [0.48], [0.02]
])

array_of_Cs1 = np.array([
    [0.25], [0.25], [0.48], [0.02]
])

Cs = np.array([array_of_Cs, array_of_Cs1])
means = np.array([array_of_gaussian_means, array_of_gaussian_means1])
covs = np.array([array_of_gaussian_covs, array_of_gaussian_covs1])

X_mfcc = np.array(
[
    [1, 1],
    [1, 1],
    [2, 2],
    [1.5, 1.5],
    [1, 2]
]
)