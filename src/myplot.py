import matplotlib.pyplot as plt
import numpy as np

def makeplot(AllData, plotvars, prediction, plotname):
    Nobs = len(AllData["observables"][0][1])
    figure, axes = plt.subplots(figsize = (3*Nobs, 5), ncols = Nobs, nrows = 2)

    for s2 in range(0, Nobs):
        axes[0][s2].set_title(AllData["observables"][0][1][s2])
        axes[0][s2].set_ylabel(plotvars[s2][1])
        axes[1][s2].set_xlabel(plotvars[s2][0])
        axes[1][s2].set_ylabel(r"ratio")
        
        S1 = AllData["systems"][0]
        O  = AllData["observables"][0][0]
        S2 = AllData["observables"][0][1][s2]
        
        DX = AllData["data"][S1][O][S2]['x']
        DY = AllData["data"][S1][O][S2]['y']
        DE = np.sqrt(AllData["data"][S1][O][S2]['yerr']['stat'][:,0]**2 + AllData["data"][S1][O][S2]['yerr']['sys'][:,0]**2)
                
        if plotname is 'Priors':
            linecount = len(prediction[S1][O][S2]['Y'])
            for i, y in enumerate(prediction[S1][O][S2]['Y']):
                axes[0][s2].plot(DX, y, 'b-', alpha=10/linecount, label=plotname if i==0 else '')
                axes[1][s2].plot(DX, y/DY, 'b-', alpha=10/linecount, label=plotname if i==0 else '')
        else:
            linecount = len(prediction[S1][O][S2])
            for i, y in enumerate(prediction[S1][O][S2]):
                axes[0][s2].plot(DX, y, 'b-', alpha=10/linecount, label=plotname if i==0 else '')
                axes[1][s2].plot(DX, y/DY, 'b-', alpha=10/linecount, label=plotname if i==0 else '')
        
        axes[0][s2].errorbar(DX, DY, yerr = DE, fmt='ro', label="Measurements")
        axes[1][s2].plot(DX, 1+(DE/DY), 'b-', linestyle = '--', color='red')
        axes[1][s2].plot(DX, 1-(DE/DY), 'b-', linestyle = '--', color='red')
        axes[1][s2].axhline(y = 1, linestyle = '--')
        axes[0][s2].set_xscale(plotvars[s2][2])
        axes[1][s2].set_xscale(plotvars[s2][2])
        axes[0][s2].set_yscale(plotvars[s2][3])
        axes[1][s2].set_ylim([0,2])

    plt.tight_layout()
    figure.subplots_adjust(hspace=0)
    figure.savefig('plots/'+plotname+'.pdf', dpi = 192)
    # figure