import numpy
import pandas
import math
from sklearn.cluster import AgglomerativeClustering
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
from scipy.stats import gaussian_kde
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error

############################## Danzart Model ##########################
def quadratic(alpha, X2C):
    vx = X2C[:, 0]
    vy = X2C[:, 1]
    OLc = alpha[0] + alpha[1] * vx + alpha[2] * vy + alpha[3] * vx ** 2 + alpha[4] * vy ** 2 + alpha[5] * vx * vy
    return OLc

def funerror(alpha, X2C, OL):
    OLc = quadratic(alpha, X2C)
    return 0.5 * numpy.sum((OLc - OL) ** 2)

############################## LIKING PRODUCT LANDSCAPE ###############
class LikingProductLandscape:
    def __init__(self,X,consumers_map='MDS',preference_map='SVM'):
        #consumers_map = ['IPM','PCA','MDS']
        #preference_map = ['Danzart','SVM']

        self.consumers_map = consumers_map
        self.preference_map = preference_map

        self.X = X #defined in step 1
        self.X2 = None #defined in step 1
        self.gauss_kde = None #defined in step 1

        #Internal Preference Map
        self.XP2 = None
        self.pca_components_variance = None

        #### Product and attribtues
        self.products_names = []
        self.products_values = []
        self.attributes_names = []
        self.attributes_values = []

    def products_overall_liking(self,product_names,products_overall_liking):
        self.products_names = product_names
        self.products_values = products_overall_liking

    def attribute(self,attribute_name,attribute_values):
        self.attributes_names.append(attribute_name)
        self.attributes_values.append(attribute_values)

    def _color_bar(self,jar,ol):
        if jar==True:
            m_jar = numpy.zeros((45, 3, 4))
            norm = matplotlib.colors.Normalize(vmin=1, vmax=5)
            v = 0
            for i in range(5):
                m_jar[v:v + 9, :, :] = matplotlib.cm.jet(norm(5 - i))
                v = v + 9
            if ol==True:
                plt.subplot(1, 8, 8)
            else:
                plt.subplot(1, 8, 7)
            plt.imshow(m_jar)
            plt.ylabel('Just About Right',fontsize=10)
            plt.xticks([])
            plt.yticks([])
            plt.text(0.6, 22, 'JAR', rotation=90, color='gray')
            plt.text(0.6, 7, 'too much', rotation=90, color='white')
            plt.text(0.6, 42, 'too weak', rotation=90, color='white')
        if ol==True:
            m_ol = numpy.zeros((45, 3, 4))
            norm = matplotlib.colors.Normalize(vmin=1, vmax=9)
            v = 0
            for i in range(9):
                m_ol[v:v + 5, :, :] = matplotlib.cm.jet(norm(9 - i))
                v = v + 5
            plt.subplot(1, 8, 7)
            plt.imshow(m_ol)
            plt.ylabel('Overall liking',fontsize=10)
            plt.xticks([])
            plt.yticks([])
            y = 2.5
            for i in range(9,0,-1):
                plt.text(0.6, y, str(i), color='white')
                y += 5

    def _danzart(self):
        plt.figure(facecolor='white')
        plt.subplot(1,2,1)
        self._product_acceptance_map(numpy.mean(self.products_values,axis=1), 'Overall liking average', 'ol')

        plt.subplot(1, 2, 2)
        m_ol = numpy.zeros((45, 3, 4))
        norm = matplotlib.colors.Normalize(vmin=1, vmax=9)
        v = 0
        for i in range(9):
            m_ol[v:v + 5, :, :] = matplotlib.cm.jet(norm(9 - i))
            v = v + 5
        plt.imshow(m_ol)
        plt.xticks([])
        plt.yticks([])
        y = 2.5
        for i in range(9, 0, -1):
            plt.text(0.6, y, str(i), color='white')
            y += 5
        plt.savefig(self.filename+'_ol_danzart.png')

    def _deep_analysisBORRAR(self,title,labels,values,types,c_jar=False,c_ol=False):
        print('deep analysis borrar')
        nrows = 2
        ncols = 4
        k_figures = [1,2,3,5,6,7]
        #plt.figure(facecolor='white')
        k = 0
        cont_figures = 1
        for i in range(len(labels)):
            if k >= len(k_figures):
                print('save fig')
                plt.figure()
                k = 0
                cont_figures += 1
            k += 1
            plt.figure(facecolor='white')
            plt.title(title)
            self._product_acceptance_mapBORRAR(values[:, i], labels[i], types[i])
            plt.savefig(self.filename + '_' + title + '_' + labels[i] + '_PAM.png')


            plt.figure(facecolor='white')
            plt.title(title)
            self._product_acceptance_map(values[:, i], labels[i], types[i])
            plt.savefig(self.filename + '_' + title + '_' + labels[i] + '_LPL.png')

    def _deep_analysis(self,title,labels,values,types,c_jar=False,c_ol=False):
        nrows = 2
        ncols = 4
        k_figures = [1,2,3,5,6,7]
        plt.figure(facecolor='white')
        k = 0
        cont_figures = 1
        for i in range(len(labels)):
            if k >= len(k_figures):
                plt.suptitle(title)
                plt.savefig(self.filename + '_' +title+'_'+ str(cont_figures) + '.png')
                plt.figure()
                k = 0
                cont_figures += 1
            plt.subplot(nrows, ncols, k_figures[k])
            k += 1
            error = self._product_acceptance_map(values[:, i], labels[i], types[i])
            error_file = open(self.filename+'.csv','a')
            error_file.write(title+' '+labels[i]+','+str(error)+' \n')
            error_file.close()
        plt.suptitle(title)

        # Color bar
        self._color_bar(jar=c_jar,ol=c_ol)

        plt.subplots_adjust(bottom=0.02,top=0.98,left=0.06,right=0.98)
        if cont_figures > nrows * ncols:
            plt.savefig(self.filename + '_' +title+'_'+ str(cont_figures) + '.png')
        else:
            plt.savefig(self.filename+'_'+title+'.png')

    def execute(self,filename):
        self.filename = filename
        error_file = open(self.filename+'.csv','w')
        error_file.write('Map,error \n')
        error_file.close()
        if len(self.products_names)==0:
            print('No se tienen productos registrados')
            return

        self._step1_consumers_map()
        types = []

        self._danzart()
        # deep analysis overall liking
        for i in range(len(self.products_names)):
            types.append('ol')
        self._deep_analysis('Overall_liking',self.products_names,self.products_values,types,c_ol=True)
        #self._deep_analysisBORRAR('Overall liking', self.products_names, self.products_values, types, c_ol=True)

        # deep analysis product
        types = []
        types.append('ol')
        for i in range(len(self.attributes_names)):
            types.append('jar')
        for p in range(len(self.products_names)):
            XX = numpy.reshape(self.products_values[:, p], (len(self.X2), 1))
            names = ['Overall liking']
            for a in range(len(self.attributes_names)):
                XX = numpy.concatenate((XX, numpy.reshape(self.attributes_values[a][:, p], (len(self.X2), 1))), axis=1)
                names.append(self.attributes_names[a])
            self._deep_analysis('Product_' + self.products_names[p], names, XX, types,c_ol=True,c_jar=True)

        # deep analysis attribute
        types = []
        for i in range(len(self.products_names)):
            types.append('jar')
        for a in range(len(self.attributes_names)):
            self._deep_analysis(self.attributes_names[a],self.products_names,self.attributes_values[a],types,c_jar=True)



################################################################################################################
########## 1. Consumer's map ###################################################################################
################################################################################################################
    def _step1_consumers_map(self):
        if self.consumers_map=='IPM':
            NC,NP = self.X.shape
            pca = PCA(random_state=0)
            XX = numpy.transpose(self.X)
            self.XP2 = pca.fit_transform(XX)
            x2var = numpy.var(self.XP2)
            self.X2 = numpy.zeros((NC, 2))
            self.X2[:, 0] = pca.components_[0] * x2var
            self.X2[:, 1] = pca.components_[1] * x2var
            self.pca_components_variance = numpy.round(pca.explained_variance_ratio_[:2]*100,2)
        elif self.consumers_map=='PCA':
            pca = PCA(random_state=0)
            self.X2 = pca.fit_transform(self.X)
            self.X2 = self.X2[:,:2]
            self.pca_components_variance = numpy.round(pca.explained_variance_ratio_[:2]*100,2)
        else: #MDS
            mds = MDS(random_state=0)
            self.X2 = mds.fit_transform(self.X)
        self.gauss_kde = gaussian_kde(self.X2.transpose())
        self.print_consumers_map()
        self.print_consumers_mapBORRAR()

    def _min_max_xy(self):
        minx = min([min(self.X2[:, 0]), min(self.X2[:, 0])]) - 5
        miny = min([min(self.X2[:, 1]), min(self.X2[:, 1])]) - 5
        maxx = max([max(self.X2[:, 0]), max(self.X2[:, 0])]) + 5
        maxy = max([max(self.X2[:, 1]), max(self.X2[:, 1])]) + 5
        if self.consumers_map=='IPM':
            minxP = min([min(self.XP2[:, 0]), min(self.XP2[:, 0])]) - 5
            minyP = min([min(self.XP2[:, 1]), min(self.XP2[:, 1])]) - 5
            maxxP = max([max(self.XP2[:, 0]), max(self.XP2[:, 0])]) + 10
            maxyP = max([max(self.XP2[:, 1]), max(self.XP2[:, 1])]) + 10
            return min(minx,minxP),min(miny,minyP),max(maxx,maxxP),max(maxy,maxyP)
        else:
            return minx,miny,maxx,maxy

    def print_consumers_mapBORRAR(self,title='',consumers_labels=[],consumers_colors=[]):
        minx, miny, maxx, maxy = self._min_max_xy()

        # consumers' distribution
        rowx = numpy.linspace(minx, maxx, 100)
        rowy = numpy.linspace(miny, maxy, 100)
        xx, yy = numpy.meshgrid(rowx, rowy)
        zz = numpy.zeros((100, 100), float)
        aa = numpy.zeros((100, 100), float)
        XY = numpy.zeros((1, 2), float)
        for x in range(100):
            for y in range(100):
                XY[0, 0] = xx[x, y]
                XY[0, 1] = yy[x, y]
                aa[x, y] = self.gauss_kde(XY)

        plt.figure()
        plt.xlim([minx, maxx])
        plt.ylim([miny, maxy])
        plt.xticks([])
        plt.yticks([])
        #if len(consumers_labels)==0:
        #    plt.contourf(xx,yy,aa,cmap='gray')
        #else:
        #    plt.contour(xx,yy,aa,cmap='gray')

        if self.consumers_map=='IPM':  # IntPrefMap
            plt.title('Internal Preference Map')
            plt.xlabel(str(self.pca_components_variance[0]) + '%')
            plt.ylabel(str(self.pca_components_variance[1]) + '%')
            plt.plot([minx, maxx], [0, 0], color='gray')
            plt.plot([0, 0], [miny, maxy], color='gray')
            for i in range(len(self.XP2)):
                plt.text(self.XP2[i, 0], self.XP2[i, 1], self.products_names[i], color='blue')
        elif self.consumers_map=='PCA':
            plt.title('Principal Component Analysis')
            plt.xlabel(str(self.pca_components_variance[0]) + '%')
            plt.ylabel(str(self.pca_components_variance[1]) + '%')
        else:
            plt.title('MultiDimensional Scaling ')
            plt.xlabel('Stress ' + str(round(self._stress(self.X, self.X2), 2)))

        if len(consumers_labels)==0:
            plt.plot(self.X2[:, 0], self.X2[:, 1], 'o', color='black')
            #for i in range(len(self.X2)):
            #    plt.text(self.X2[i, 0], self.X2[i, 1], '.',color='black')
            plt.savefig(self.filename + '_consumers_map2.png')
        else:
            for i in range(len(self.X2)):
                plt.text(self.X2[i, 0], self.X2[i, 1],consumers_labels[i],color=consumers_colors[i])
            plt.savefig(self.filename + '_consumers_map2_'+title+'.png')

    def print_consumers_map(self,title='',consumers_labels=[],consumers_colors=[]):
        minx, miny, maxx, maxy = self._min_max_xy()

        # consumers' distribution
        rowx = numpy.linspace(minx, maxx, 100)
        rowy = numpy.linspace(miny, maxy, 100)
        xx, yy = numpy.meshgrid(rowx, rowy)
        zz = numpy.zeros((100, 100), float)
        aa = numpy.zeros((100, 100), float)
        XY = numpy.zeros((1, 2), float)
        for x in range(100):
            for y in range(100):
                XY[0, 0] = xx[x, y]
                XY[0, 1] = yy[x, y]
                aa[x, y] = self.gauss_kde(XY)

        plt.figure()
        plt.xlim([minx, maxx])
        plt.ylim([miny, maxy])
        plt.xticks([])
        plt.yticks([])
        if len(consumers_labels)==0:
            plt.contourf(xx,yy,aa,cmap='gray')
        else:
            plt.contour(xx,yy,aa,cmap='gray')

        if self.consumers_map=='IPM':  # IntPrefMap
            plt.title('Internal Preference Map')
            plt.xlabel(str(self.pca_components_variance[0]) + '%')
            plt.ylabel(str(self.pca_components_variance[1]) + '%')
            plt.plot([minx, maxx], [0, 0], color='gray')
            plt.plot([0, 0], [miny, maxy], color='gray')
            for i in range(len(self.XP2)):
                plt.text(self.XP2[i, 0], self.XP2[i, 1], self.products_names[i], color='gray')
        elif self.consumers_map=='PCA':
            plt.title('Principal Component Analysis')
            plt.xlabel(str(self.pca_components_variance[0]) + '%')
            plt.ylabel(str(self.pca_components_variance[1]) + '%')
        else:
            plt.title('MultiDimensional Scaling ')
            plt.xlabel('Stress ' + str(round(self._stress(self.X, self.X2), 2)))

        if len(consumers_labels)==0:
            for i in range(len(self.X2)):
                plt.text(self.X2[i, 0], self.X2[i, 1], '.',color='blue')
            plt.savefig(self.filename + '_consumers_map.png')
        else:
            for i in range(len(self.X2)):
                plt.text(self.X2[i, 0], self.X2[i, 1],consumers_labels[i],color=consumers_colors[i])
            plt.savefig(self.filename + '_consumers_map_'+title+'.png')

    def _stress_distance_matrix(self,X):
        N = len(X)
        D = numpy.zeros((N, N), float)
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                D[i, j] = D[j, i] = numpy.linalg.norm(X[i, :] - X[j, :])
        return D

    def _stress(self,X, X2):
        D = self._stress_distance_matrix(X)
        D2 = self._stress_distance_matrix(X2)
        v1 = 0
        v2 = 0
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                v1 += (D[i, j] - D2[i, j]) ** 2
                v2 += D[i, j] ** 2
        return math.sqrt(v1 / v2)

################################################################################################################
########## 2. Consumer's groups ###################################################################################
################################################################################################################

    def step2_consumer_groups(self,ngroup,demographic_labels,demographic_data,demographic_data_discrete,filename):
        ag = AgglomerativeClustering(n_clusters=ngroup, affinity='euclidean', linkage='ward')
        cluster = ag.fit_predict(self.X)

        minx, miny, maxx, maxy = self._min_max_xy()

        # consumers' distribution
        rowx = numpy.linspace(minx, maxx, 100)
        rowy = numpy.linspace(miny, maxy, 100)
        xx, yy = numpy.meshgrid(rowx, rowy)
        zz = numpy.zeros((100, 100), float)
        aa = numpy.zeros((100, 100), float)
        XY = numpy.zeros((1, 2), float)
        for x in range(100):
            for y in range(100):
                XY[0, 0] = xx[x, y]
                XY[0, 1] = yy[x, y]
                aa[x, y] = self.gauss_kde(XY)

        plt.figure()
        plt.xlim([minx, maxx])
        plt.ylim([miny, maxy])
        plt.xticks([])
        plt.yticks([])
        plt.contourf(xx, yy, aa, cmap='gray')

        if self.consumers_map == 'IPM':  # IntPrefMap
            plt.title('IPM - Consumer Segments')
            plt.xlabel(str(self.pca_components_variance[0]) + '%')
            plt.ylabel(str(self.pca_components_variance[1]) + '%')
            #plt.plot([minx, maxx], [0, 0], color='black')
            #plt.plot([0, 0], [miny, maxy], color='black')
            for i in range(len(self.XP2)):
                plt.text(self.XP2[i, 0], self.XP2[i, 1], self.product_names[i], color='blue')
        elif self.consumers_map=='PCA':
            plt.title('PCA - Consumer Segments')
            plt.xlabel(str(self.pca_components_variance[0]) + '%')
            plt.ylabel(str(self.pca_components_variance[1]) + '%')
        else:
            plt.title('MDS - Consumer Segments ')

        for i in range(len(self.X2)):
            plt.text(self.X2[i, 0], self.X2[i, 1], str(cluster[i]), color='gray')
        plt.savefig(filename)

        file = open(filename+'.txt','w')
        groups, group_count = numpy.unique(cluster, return_counts=True)
        for g in groups:
            file.write('----- Group '+str(g)+'-----\n')

            for i in range(len(demographic_data)):
                file.write(demographic_labels[i]+':\n')
                if demographic_data_discrete[i]==1:
                    values,nvalues = numpy.unique(demographic_data[i][cluster==g], return_counts=True)
                    for v in range(len(values)):
                        per = round(nvalues[v]*100/sum(cluster==g),1)
                        file.write('  '+str(values[v])+' '+str(per)+'%\n')
                else:
                    file.write('  mean:'+str(round(numpy.mean(demographic_data[i][cluster==g]),2))+'std:'+str(round(numpy.std(demographic_data[i][cluster==g]),2))+'\n')
        file.close()

################################################################################################################
########## Liking Product Landscape ###################################################################################
################################################################################################################
    def _product_acceptance_mapBORRAR(self,G,title,type='OL'):
        minx, miny, maxx, maxy = self._min_max_xy()
        plt.title(title)
        plt.xlim([minx, maxx])
        plt.ylim([miny, maxy])
        plt.xticks([])
        plt.yticks([])
        svr = SVR()
        if self.preference_map == 'Danzart':
            alpha = [1, 1, 1, 1, 1, 1, 1, 1]
            bounds = numpy.zeros((len(alpha), 2))
            bounds[:,0] = -5
            bounds[:,1] = 5
            de = differential_evolution(funerror, bounds, args=(self.X2, G),seed=0)
            alpha = de.x
            G_ = quadratic(alpha, self.X2)
        else:
            svr.fit(self.X2, G)
            G_ = svr.predict(self.X2)

        error = math.sqrt(mean_squared_error(G_,G))
        #error = numpy.mean(numpy.abs(G_ - G))
        if type == 'jar':
            errorn = (error*100)/5
        else:
            errorn = (error*100)/9
        plt.ylabel('Error '+str(round(errorn,1))+'%',fontsize=8)

        rowx = numpy.linspace(minx, maxx, 100)
        rowy = numpy.linspace(miny, maxy, 100)
        xx, yy = numpy.meshgrid(rowx, rowy)
        zz = numpy.zeros((100, 100), float)
        aa = numpy.zeros((100, 100), float)
        XY = numpy.zeros((1, 2), float)
        maxpdf = 0
        for x in range(100):
            for y in range(100):
                XY[0, 0] = xx[x, y]
                XY[0, 1] = yy[x, y]
                if self.preference_map == 'Danzart':
                    zz[x, y] = quadratic(alpha, XY)
                else:
                    zz[x, y] = svr.predict(XY)
                aa[x, y] = self.gauss_kde(XY)
                if aa[x, y] > maxpdf:
                    maxpdf = aa[x, y]

        if type == 'jar':
            color = numpy.zeros((100, 110, 4), float)
            n = len(G)
            cont = numpy.zeros((5), float)
            cont[0] = sum(G <= -1.5) / n  # -2.5,-1.5
            cont[1] = sum(numpy.logical_and(-1.5 < G, G <= -0.5)) / n  # -1.5,-.5
            cont[2] = sum(numpy.logical_and(-0.5 < G, G <= 0.5)) / n  # -.5,.5
            cont[3] = sum(numpy.logical_and(0.5 < G, G <= 1.5)) / n  # .5,1.5
            cont[4] = sum(1.5 < G) / n  # 1.5,2.5
            jar = round(cont[2]*100,1)
            cont = numpy.cumsum(cont)
            cont = (cont*n).astype(int)
            cont_v = [-2, -1, 0, 1, 2]
            c = 0
            for y in range(100):
                for x in range(100,110):
                    norm = matplotlib.colors.Normalize(vmin=-2, vmax=2)
                    color[99-y, x,:] = matplotlib.cm.jet(norm(cont_v[c]))
                if y == cont[c]:
                    c += 1
            plt.xlabel('JAR '+str(jar)+'%',fontsize=8)
        else:
            color = numpy.zeros((100, 100, 4), float)
            vmean = round(numpy.mean(G),1)
            plt.xlabel('Average '+str(vmean),fontsize=8)
        for x in range(100):
            for y in range(100):
                if type == 'jar':
                    norm = matplotlib.colors.Normalize(vmin=-2, vmax=2)
                    c = list(matplotlib.cm.jet(norm(zz[x, y])))
                else:
                    norm = matplotlib.colors.Normalize(vmin=1, vmax=9)
                    c = list(matplotlib.cm.jet(norm(zz[x, y])))
                #c[3] = aa[x, y] / maxpdf
                color[99 - x, y, :] = c
        color = (color * 255).astype(int)

        if self.consumers_map == 'IPM':  # IntPrefMap
            for j in range(len(self.XP2)):
                plt.text(self.XP2[j, 0], self.XP2[j, 1], self.products_names[j], color='blue')

        plt.imshow(color, extent=(minx, maxx, miny, maxy))
        #plt.contour(xx, yy, aa, cmap='gray', linewidths=0.5)
        # plt.contourf(xx,yy,zz,vmin=1,vmax=9,cmap='jet')

        for j in range(len(self.X2)):
            plt.scatter(self.X2[j, 0], self.X2[j, 1], cmap='jet', c=G[j],vmin=1, vmax=9)
            #plt.text(self.X2[j, 0], self.X2[j, 1], '.', color='gray')

    def _product_acceptance_map(self,G,title,type='OL'):
        minx, miny, maxx, maxy = self._min_max_xy()
        plt.title(title)
        plt.xlim([minx, maxx])
        plt.ylim([miny, maxy])
        plt.xticks([])
        plt.yticks([])
        svr = SVR()
        if self.preference_map == 'Danzart':
            alpha = [1, 1, 1, 1, 1, 1, 1, 1]
            bounds = numpy.zeros((len(alpha), 2))
            bounds[:,0] = -5
            bounds[:,1] = 5
            de = differential_evolution(funerror, bounds, args=(self.X2, G),seed=0)
            alpha = de.x
            G_ = quadratic(alpha, self.X2)
        else:
            svr.fit(self.X2, G)
            G_ = svr.predict(self.X2)

        error = math.sqrt(mean_squared_error(G_,G))
        #error = numpy.mean(numpy.abs(G_ - G))
        if type == 'jar':
            errorn = (error*100)/5
        else:
            errorn = (error*100)/9
        plt.ylabel('Error '+str(round(errorn,1))+'%',fontsize=8)

        rowx = numpy.linspace(minx, maxx, 100)
        rowy = numpy.linspace(miny, maxy, 100)
        xx, yy = numpy.meshgrid(rowx, rowy)
        zz = numpy.zeros((100, 100), float)
        aa = numpy.zeros((100, 100), float)
        XY = numpy.zeros((1, 2), float)
        maxpdf = 0
        for x in range(100):
            for y in range(100):
                XY[0, 0] = xx[x, y]
                XY[0, 1] = yy[x, y]
                if self.preference_map == 'Danzart':
                    zz[x, y] = quadratic(alpha, XY)
                else:
                    zz[x, y] = svr.predict(XY)
                aa[x, y] = self.gauss_kde(XY)
                if aa[x, y] > maxpdf:
                    maxpdf = aa[x, y]

        if type == 'jar':
            color = numpy.zeros((100, 110, 4), float)
            n = len(G)
            cont = numpy.zeros((5), float)
            cont[0] = sum(G <= -1.5) / n  # -2.5,-1.5
            cont[1] = sum(numpy.logical_and(-1.5 < G, G <= -0.5)) / n  # -1.5,-.5
            cont[2] = sum(numpy.logical_and(-0.5 < G, G <= 0.5)) / n  # -.5,.5
            cont[3] = sum(numpy.logical_and(0.5 < G, G <= 1.5)) / n  # .5,1.5
            cont[4] = sum(1.5 < G) / n  # 1.5,2.5
            jar = round(cont[2]*100,1)
            cont = numpy.cumsum(cont)
            cont = (cont*100).astype(int)
            cont_v = [-2, -1, 0, 1, 2]
            c = 0
            for y in range(100):
                for x in range(100,110):
                    norm = matplotlib.colors.Normalize(vmin=-2, vmax=2)
                    color[99-y, x,:] = matplotlib.cm.jet(norm(cont_v[c]))
                if y == cont[c]:
                    c += 1
            plt.xlabel('JAR '+str(jar)+'%',fontsize=8)
        else:
            color = numpy.zeros((100, 100, 4), float)
            vmean = round(numpy.mean(G),1)
            plt.xlabel('Average '+str(vmean),fontsize=8)
        for x in range(100):
            for y in range(100):
                if type == 'jar':
                    norm = matplotlib.colors.Normalize(vmin=-2, vmax=2)
                    c = list(matplotlib.cm.jet(norm(zz[x, y])))
                else:
                    norm = matplotlib.colors.Normalize(vmin=1, vmax=9)
                    c = list(matplotlib.cm.jet(norm(zz[x, y])))
                c[3] = aa[x, y] / maxpdf
                color[99 - x, y, :] = c
        color = (color * 255).astype(int)

        if self.consumers_map == 'IPM':  # IntPrefMap
            for j in range(len(self.XP2)):
                plt.text(self.XP2[j, 0], self.XP2[j, 1], self.products_names[j], color='blue')

        plt.imshow(color, extent=(minx, maxx, miny, maxy))
        plt.contour(xx, yy, aa, cmap='gray', linewidths=0.5)
        # plt.contourf(xx,yy,zz,vmin=1,vmax=9,cmap='jet')
        for j in range(len(self.X2)):
            plt.text(self.X2[j, 0], self.X2[j, 1], '.', color='gray')
        return errorn

################################################################################################################
########## Methodology ###################################################################################
################################################################################################################

    def methodology_print_acceptancemap_construction(self,filename):
        minx, miny, maxx, maxy = self._min_max_xy()
        fig = plt.figure(facecolor="white")
        fig.suptitle('Product acceptance map construction')

        G = self.X[:, 1]
        TMG = 100
        norm = matplotlib.colors.Normalize(vmin=1, vmax=9)
        #------------- Map 1. Consumers points ------------------------
        ax = fig.add_subplot(1, 4, 1, projection='3d')
        plt.xlim([minx, maxx])
        plt.ylim([miny, maxy])
        plt.xticks([])
        plt.yticks([])
        for i in range(len(self.X2)):
            ax.scatter(self.X2[i, 0], self.X2[i, 1], zs=0, zdir='z',color=plt.cm.jet(norm(G[i])))
        ax.set_zlim(0, 10)
        ax.set_title("a)")

        # ------------- Map 2. Consumers grades sticks -----------------
        ax = fig.add_subplot(1, 4, 2, projection='3d')
        plt.xlim([minx, maxx])
        plt.ylim([miny, maxy])
        plt.xticks([])
        plt.yticks([])
        ax.scatter(self.X2[:, 0], self.X2[:, 1], zs=0, zdir='z')
        ax.set_zlim(0, 10)
        for i in range(len(self.X2)):
            x = [self.X2[i, 0], self.X2[i, 0]]
            y = [self.X2[i, 1], self.X2[i, 1]]
            z = [0, G[i]]
            # ax.text(X[i,0])
            ax.plot(x, y, z, color=plt.cm.jet(norm(G[i])))
        ax.set_title("b)")

        # ------------- Map 3. Consumers grades model 3d -----------------
        svr = SVR()
        if self.preference_map == 'Danzart':
            alpha = [1, 1, 1, 1, 1, 1, 1, 1]
            bounds = numpy.zeros((len(alpha), 2))
            bounds[:, 0] = -5
            bounds[:, 1] = 5
            de = differential_evolution(funerror, bounds, args=(self.X2, G),seed=0)
            alpha = de.x
            G_ = quadratic(alpha, self.X2)
        else:
            svr.fit(self.X2, G)
            G_ = svr.predict(self.X2)

        rowx = numpy.linspace(minx, maxx, TMG)
        rowy = numpy.linspace(miny, maxy, TMG)
        xx, yy = numpy.meshgrid(rowx, rowy)
        zz = numpy.zeros((TMG, TMG), float)
        aa = numpy.zeros((TMG, TMG), float)
        XY = numpy.zeros((1, 2), float)
        maxpdf = 0
        for x in range(TMG):
            for y in range(TMG):
                XY[0, 0] = xx[x, y]
                XY[0, 1] = yy[x, y]
                if self.preference_map == 'Danzart':
                    zz[x, y] = quadratic(alpha,XY)
                else:
                    zz[x, y] = svr.predict(XY)
                aa[x, y] = self.gauss_kde(XY)
                if aa[x, y] > maxpdf:
                    maxpdf = aa[x, y]

        norm = matplotlib.colors.Normalize(vmin=1, vmax=9)
        lpl = numpy.zeros((TMG, TMG, 4), float)
        color = numpy.zeros((TMG, TMG, 4), float)
        for x in range(TMG):
            for y in range(TMG):
                c = list(matplotlib.cm.jet(norm(zz[x, y])))
                color[TMG - 1 - x, y, :] = numpy.copy(c)
                c[3] = aa[x, y] / maxpdf
                lpl[TMG - 1 - x, y, :] = c
        lpl = (lpl * 255).astype(int)
        color = (color*255).astype(int)

        ax = fig.add_subplot(1, 4, 3, projection='3d')
        plt.xlim([minx, maxx])
        plt.ylim([miny, maxy])
        plt.xticks([])
        plt.yticks([])
        ax.plot_surface(xx, yy, zz, facecolors=plt.cm.jet(norm(zz)), shade=False)
        for i in range(len(self.X2)):
            ax.scatter(self.X2[i, 0], self.X2[i, 1], zs=0, zdir='z', color=plt.cm.jet(norm(G[i])))

        ax.set_title("c)")
        # ------------- Map 4. Consumers grades model 2d -----------------
        fig.add_subplot(1, 4, 4)
        plt.imshow(color, extent=(minx, maxx, miny, maxy))
        plt.xticks([])
        plt.yticks([])
        for i in range(len(self.X2)):
            plt.text(self.X2[i, 0], self.X2[i, 1], '.', color='gray')


        plt.title("d)")

        plt.show()

        ############ Figura 2 ###################################
        fig = plt.figure(facecolor="white")

        # ------------- Map 1. Consumers grades model 2d -----------------
        fig.add_subplot(1, 3, 2)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(color, extent=(minx, maxx, miny, maxy))
        for i in range(len(self.X2)):
            plt.text(self.X2[i, 0], self.X2[i, 1], '.', color='gray')
        plt.title("b)")

        # ------------- Map 2. Consumers distribution map -----------------
        fig.add_subplot(1, 3, 1)
        plt.xlim([minx, maxx])
        plt.ylim([miny, maxy])
        plt.xticks([])
        plt.yticks([])
        plt.contourf(xx, yy, aa, cmap='gray')
        plt.title("a)")

        # ------------- Map 3. Mix distribution and liking -----------------
        fig.add_subplot(1, 3, 3)
        plt.title("c)")
        plt.imshow(lpl, extent=(minx, maxx, miny, maxy))
        plt.xticks([])
        plt.yticks([])
        plt.contour(xx, yy, aa, cmap='gray', linewidths=0.5)
        # plt.contourf(xx,yy,zz,vmin=1,vmax=9,cmap='jet')
        for i in range(len(self.X2)):
            plt.text(self.X2[i, 0], self.X2[i, 1], '.', color='gray')

        plt.show()

        ############ Figura 3 ###################################
        if self.preference_map == 'Danzart':
            G_ = quadratic(alpha, self.X2)
        else:
            G_ = svr.predict(self.X2)
        error = math.sqrt(mean_squared_error(G,G_))
        #error = numpy.mean(numpy.abs(G_ - G))
        error = round(error, 2)

        plt.figure(facecolor="white")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(color, extent=(minx, maxx, miny, maxy))
        for i in range(len(self.X2)):
            plt.scatter(self.X2[i, 0], self.X2[i, 1], color=plt.cm.jet(norm(G[i])))
        plt.title('Error:'+str(round(error,2)))

        plt.show()



