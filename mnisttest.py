from __future__ import print_function
from utils import plott
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from utils.netfitter import NetworkFitter
from sklearn.datasets import make_gaussian_quantiles
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from sklearn.svm import SVR
import pandas as pd
import os

df = pd.DataFrame(columns=['iteration','net_size','nin','nout','datasetsize','nettype','k','p','eq_time','minmax_volt','rnmn_score', 'log_orig_score','svr_score','circuit'])

def main():
    # ttables = {}
    # ttables['xor'] = [[-1, -1, 0], [-1, 1, 1], [1, -1, 1], [1, 1, 0]]
    # ttables['or'] = [[-1, -1, 0], [-1, 1, 1], [1, -1, 1], [1, 1, 0]]
    # ttables['and'] = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]

    nin=2
    nout=30

    nf = NetworkFitter()

    circ=nf.generate_random_net_circuit(n=40,nin=nin,nout=nout,k=5,p=0.1)
    plott.plot_json_graph(circ['circuit'])
    nf.circuit=circ
    nf.eq_time=0.01

    # data=np.array(ttables['xor']*1)
    # X = data[:,:-1]
    # y = data[:,-1]

    X_orig, y = make_gaussian_quantiles(n_features=nin, n_classes=2,  n_samples=80)
    X = X_orig
    X = np.multiply(X, 30)

    resx = nf.network_eval(X, y)

    # X= [[-0.00010509833242274752, -2.5950890802248667e-05, -0.002232547794357209, 0.0], [-4.561918433269682e-06, -5.083452580674139e-06, -9.586250491245837e-05, 1.0], [2.6346789297937676e-05, 0.002092832614595492, -0.00011734170206765503, 1.0], [0.00019287569796774096, 0.002305312827739898, 0.00034009405688103264, 0.0], [-0.0001278893414836002, -3.152015355331398e-05, -0.0027176867965719876, 0.0], [-3.958255704354952e-06, -4.614759904728698e-06, -8.549782461503566e-05, 1.0], [2.0179714762362603e-05, 0.0018475790758137088, -0.0001252032043363086, 1.0], [0.0001536547088920122, 0.0017381644560382032, 0.00027327172431839406, 0.0], [-0.0001189521035443556, -2.8885491973419172e-05, -0.0025394655190599407, 0.0], [-4.973314806884533e-06, -5.229967907062977e-06, -0.00010095772709088284, 1.0], [2.678329401810996e-05, 0.0020267100284509414, -0.00010337262854586885, 1.0], [0.00015928292861900158, 0.001751713404201791, 0.0002862666029788268, 0.0], [-0.00012264413642676926, -2.9839649703446503e-05, -0.0026177365901176725, 0.0], [-7.396806363810349e-06, -6.5931336518800255e-06, -0.00013511145678663186, 1.0], [2.2728411181107493e-05, 0.0019115859105039209, -0.0001165601651564499, 1.0], [0.00017096475988864147, 0.001899526571034626, 0.00030604610464541595, 0.0], [-0.00013679549197270598, -3.3275101532710864e-05, -0.0029198579525447034, 0.0], [-2.1869996359259477e-06, -3.552780182123903e-06, -5.869236737933821e-05, 1.0], [1.6587325105452242e-05, 0.0017543733262649938, -0.00013695409361751316, 1.0], [0.0001852746691240595, 0.002032231289605188, 0.0003333141603641538, 0.0], [-0.00014234568898390323, -3.456748185464069e-05, -0.003038874763467836, 0.0], [-1.3918373626164326e-06, -3.0464892291566166e-06, -4.7862624446630767e-05, 1.0], [1.971414108901354e-05, 0.0018502464421720948, -0.00012885583146995066, 1.0], [0.00018649708026223359, 0.0021536617762156697, 0.00033066911351655094, 0.0], [-0.00012044908415648308, -2.973071048745109e-05, -0.002558817593223075, 0.0], [-3.982650078694191e-06, -4.5141568904390155e-06, -8.456815479683234e-05, 1.0], [1.8650203440653913e-05, 0.0018329540859225773, -0.00013382523644095673, 1.0], [0.00016934481874264382, 0.0019991159638365593, 0.00029920538584079647, 0.0], [-0.00011909582825314799, -2.9125372912499176e-05, -0.002537438798706846, 0.0], [-4.6192642194300554e-06, -5.394791429257693e-06, -9.991932532843423e-05, 1.0], [3.139132869091992e-05, 0.0021560932791233455, -9.049208198327335e-05, 1.0], [0.00017820265730966766, 0.0019350070964442182, 0.0003218262768347768, 0.0], [-0.00012464565008261507, -3.0279834065538555e-05, -0.0026609029487712904, 0.0], [-4.2597598392624735e-06, -4.612119748627316e-06, -8.79614027435218e-05, 1.0], [2.4977930905013432e-05, 0.002091116551439551, -0.00012670055809518548, 1.0], [0.00016116987334104034, 0.0018003398786418485, 0.00028790640700197893, 0.0], [-0.00011357231138133774, -2.7865866173503705e-05, -0.002415810073351221, 0.0], [-3.120924369605408e-06, -4.125129195664746e-06, -7.276984721186679e-05, 1.0], [1.9968475194081484e-05, 0.0016582792138430116, -9.934724396549332e-05, 1.0], [0.00015365610723703992, 0.0016579454553419925, 0.00027815751206499497, 0.0], [-0.00011276576496767851, -2.7396293164424723e-05, -0.0024072713113920323, 0.0], [-1.4427972033388806e-06, -3.109057979781389e-06, -4.986237232822574e-05, 1.0], [1.860558288927516e-05, 0.0018614888969511023, -0.00013825934594231018, 1.0], [0.000180981728325197, 0.0021659009424783056, 0.0003190552228269291, 0.0], [-0.0001096166770710619, -2.679473160864824e-05, -0.0023358674341136866, 0.0], [-6.584651278964315e-06, -6.3400947931226365e-06, -0.00012648636887355554, 1.0], [2.897091135790878e-05, 0.0021310489503421318, -0.0001032676329666913, 1.0], [0.0001641852243875243, 0.0017957307357789937, 0.0002956988825541535, 0.0], [-0.00013142001424137838, -3.217547280302177e-05, -0.002798885248813457, 0.0], [-5.090227163943377e-06, -5.581495475217175e-06, -0.00010584935688423577, 1.0], [2.1514437215013303e-05, 0.0019268135794533866, -0.00012727903008947814, 1.0], [0.0001766735056344564, 0.00196013408551703, 0.0003164426049798808, 0.0], [-0.00011210781336708987, -2.7422786827882135e-05, -0.0023883542403145015, 0.0], [-5.708820874807183e-06, -5.5724603247972775e-06, -0.0001106796259362682, 1.0], [2.9097446069464013e-05, 0.0021092711815541856, -9.938674175873389e-05, 1.0], [0.00017242450950382556, 0.001944495275328798, 0.0003068526690095085, 0.0], [-0.00013645903344594807, -3.3329232832866437e-05, -0.0029087019594430854, 0.0], [-3.7968448762230754e-06, -4.729397321026033e-06, -8.51492363387594e-05, 1.0], [1.446380244462365e-05, 0.0015606672925926699, -0.0001238892933417496, 1.0], [0.0001941469978069421, 0.002309537779584525, 0.0003426008106387823, 0.0], [-0.00011529709785216024, -2.7997980294084754e-05, -0.0024614357009726017, 0.0], [-3.5848747182247053e-06, -4.536501415589446e-06, -8.120481162219841e-05, 1.0], [2.573570425169386e-05, 0.0020023708143995953, -0.00010856593620511438, 1.0], [0.00018016669991914927, 0.002080879987090377, 0.0003194372544021531, 0.0], [-0.00011640158834038688, -2.9141128595602795e-05, -0.0024657962872964402, 0.0], [-3.474194478303369e-06, -3.840532661189853e-06, -6.981043954667888e-05, 1.0], [1.1950625782984856e-05, 0.0015011360834860398, -0.00013296159114739342, 1.0], [0.00016381440359897252, 0.0018918294279767734, 0.0002904489656115072, 0.0], [-0.00011823953370158425, -2.874401547885007e-05, -0.00252395248711207, 0.0], [-3.9528654482995235e-06, -4.582401456886355e-06, -8.456992702635803e-05, 1.0], [1.311499456993551e-05, 0.0015361097541558544, -0.00012982713132220837, 1.0], [0.00018039111941827774, 0.001976574707976667, 0.00032465985196750423, 0.0], [-0.00011575784957503619, -2.8122890990516006e-05, -0.002471148045062252, 0.0], [-1.6955498886061607e-06, -3.5890757190970808e-06, -5.96282734164195e-05, 1.0], [1.1961698568046387e-05, 0.0015226036204320185, -0.00013598740159763854, 1.0], [0.00016013897146040392, 0.0018642269083745723, 0.00028357358748870234, 0.0], [-0.0001346390277865925, -3.299208052276821e-05, -0.002866550330497533, 0.0], [-4.599878837388589e-06, -4.818934755839483e-06, -9.267012637521764e-05, 1.0], [2.651735022247101e-05, 0.002147956465540842, -0.00012410561715428477, 1.0], [0.00017748465987800136, 0.0019539652407678854, 0.00031884854463101666, 0.0]]

    pca_plotter(resx)
    # plott.plot3d(X)

    rnmnscore = nf.logreg_fit(np.array(resx)[:,:-1],y,rescale=True)
    log_orig_score = nf.logreg_fit(X, y,rescale=True)

    #SVM regression
    svr = SVR(C=10)
    svr.fit(X_orig, y)
    svr_score=svr.score(X_orig,y)

    print("Boosted RNMN logistic score: ",rnmnscore)
    print("Original logistic score: ", log_orig_score)
    print("SVR score: ", svr_score)
    return rnmnscore, log_orig_score, svr_score, circ

def pca_plotter(input,savepath=""):
    fig = plt.figure()

    std_scaler = StandardScaler()
    std_scaler.fit(input)
    input = std_scaler.transform(input)
    X = np.array(input)[:, :-1]
    y = np.array(input)[:, -1]

    # X = np.array(input)[:, :-1]
    # y = np.array(input)[:, -1]
    features=np.shape(X)[1]

    pca = PCA(n_components=features)
    X_r = pca.fit(X).transform(X)

    # ax2 = fig.add_subplot(231)
    # ax3 = fig.add_subplot(232)
    # ax4 = fig.add_subplot(233)
    # colors = ['red', 'blue']
    # for color, i in zip(colors, [min(y), max(y)]):
    #     ax2.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color)
    #     ax3.scatter(X_r[y == i, 1], X_r[y == i, 2], color=color)
    #     ax4.scatter(X_r[y == i, 2], X_r[y == i, 0], color=color)
    # pca = PCA(n_components=features)
    # X_r = pca.fit(X).transform(X)
    #
    combs=list(combinations(range(features),2))

    #create axes
    axs=[]
    pn=int(np.ceil(np.sqrt(features)))
    for comb,feat in zip(combs,range(features)):
        axs.append(fig.add_subplot(pn,pn,feat+1))

    colors = ['red', 'blue']
    for color, i in zip(colors, [min(y), max(y)]):
        for ax, comb in zip(axs, combs):
            ax.scatter(X_r[y == i, comb[0]], X_r[y == i, comb[1]], color=color)

    if savepath!="":
        plt.savefig(savepath)
    else:
        plt.show()

def other_main():
    nf = NetworkFitter()
    nf.circuit=nf.generate_random_net()
    print(nf.circuit)

def gaussian_quantiles_test():

    rootpath = "./results/gaussian_pct_3/"

    net_size=[]

    datasetsize=60
    eq_time = 0.05
    vmult=10

    times = 10
    nins = [2,4,8]
    nouts = [5,40,55]
    net_size = [100,150,200]
    pp = [0.01, 0.1, 0.4]
    kk = [2, 3, 5, 6]


    #Brute force search

    for tme in range(times):
        for nin in nins:
            for nout in nouts:
                for nsz in net_size:
                    for p in pp:
                        for k in kk:
                            try:
                                nf = NetworkFitter()
                                circ = nf.generate_random_net_circuit(n=nsz, nin=nin, nout=nout, k=k, p=p)
                                # plott.plot_json_graph(circ['circuit'])
                                nf.circuit = circ
                                nf.eq_time = eq_time

                                X_orig, y = make_gaussian_quantiles(n_features=nin, n_classes=2, n_samples=datasetsize)
                                X = X_orig
                                X = np.multiply(X, vmult)

                                resx = nf.network_eval(X, y)
                                # pca_plotter(resx)

                                rnmnscore = nf.logreg_fit(np.array(resx)[:, :-1], y, rescale=True)
                                log_orig_score = nf.logreg_fit(X, y, rescale=True)

                                # SVM regression
                                svr = SVR(C=10)
                                svr.fit(X_orig, y)
                                svr_score = svr.score(X_orig, y)

                                print("Boosted RNMN logistic score: ", rnmnscore)
                                print("Original logistic score: ", log_orig_score)
                                print("SVR score: ", svr_score)

                                #'iteration', 'net_size', 'nin', 'nout', 'datasetsize', 'nettype', 'k', 'p', 'eq_time', 'minmax_volt', 'rnmn_score', 'log_orig_score', 'svr_score', 'circuit'
                                df.loc[len(df)] = [tme,nsz, nin, nout, datasetsize, 'ws',k,p,eq_time,vmult,rnmnscore,log_orig_score,svr_score,circ]
                                df.to_csv(os.path.join(rootpath,"experiment_magicK_cusolver_commit67fee1a1.csv"),sep='|')
                            except:
                                pass

if __name__ == "__main__":
    # other_main()
    # main()
    gaussian_quantiles_test()