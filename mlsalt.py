import matplotlib.pyplot as plt
import numpy as np
import math
from numpy import trapz
from scipy.optimize import minimize
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import multivariate_normal


##
# X: 2d array with the input features
# y: 1d array with the class labels (0 or 1)
#

def robust(x):
    ind1 = np.where(x==0.0)[0]
    ind2 = np.where(x==1.0)[0]
    x[ind1] = math.pow(0.1, 9)
    x[ind2] = 1 - math.pow(0.1, 9)  
    return x


def plot_data_internal (X, y):
	x_min , x_max = X[ : , 0 ]. min () - .5 , X[ : , 0 ]. max () + .5
	y_min , y_max = X[ : , 1 ]. min () - .5 , X[ : , 1 ]. max () + .5
	xx , yy = np. meshgrid (np. linspace (x_min , x_max , 100) , \
	np. linspace (y_min , y_max , 100))
	plt. figure ()
	plt. xlim (xx.min () , xx. max ())
	plt. ylim (yy.min () , yy. max ())
	ax = plt. gca ()
	ax. plot (X[y == 0 , 0] , X[y == 0 , 1] , 'r. ', label = 'Class 1')
	ax. plot (X[y == 1 , 0] , X[y == 1 , 1] , 'b. ', label = 'Class 2')
	plt. xlabel ('X1 ')
	plt. ylabel ('X2 ')
	plt. title ('Plot data ')
	plt. legend (loc = 'upper left ', scatterpoints = 1 , numpoints = 1)
	return xx , yy

##
# X: 2d array with the input features
# y: 1d array with the class labels (0 or 1)
#
def plot_data (X, y):
	xx , yy = plot_data_internal (X, y)
	plt. show ()
	
##
# ll: 1d array with the average likelihood per data point
#
def plot_ll (ll ):
	plt. figure ()
	ax = plt. gca ()
	plt. xlim (0 , len(ll) + 2)
	plt. ylim (min(ll) - 0.1 , max(ll) + 0.1)
	ax. plot (np. arange (1 , len(ll) + 1) , ll , 'r-')
	plt. xlabel ('Steps ')
	plt. ylabel ('Average log - likelihood ')
	plt. title ('Plot Average Log - likelihood Curve ')
	plt. show ()
	
def plot_ll_color (ll, id ):
    plt.figure()
    ax = plt. gca ()
    plt. xlim (0 , len(ll) + 2)
    plt. ylim(min(ll) - 0.1 , max(ll) + 0.1)
    if id == 0:
        ax. plot (np. arange (1 , len(ll) + 1) , ll , 'b-')
        plt. title ('Plot Average Log - likelihood Curve for Training ')
    elif id == 1:
        ax. plot (np. arange (1 , len(ll) + 1) , ll , 'r-')
        plt. title ('Plot Average Log - likelihood Curve for Test')
    plt. xlabel ('Steps ')
    plt. ylabel ('Average log - likelihood ') 
    plt. show ()
    


##
# l: hyper - parameter for the width of the Gaussian bassis functions
# Z: location of the Gaussian basis functions
# X: points at which to evaluate the basis functions
def expand_inputs (l, X, Z):
	X2 = np. sum (X**2 , 1)
	Z2 = np. sum (Z**2 , 1)
	ones_Z = np. ones (Z. shape [ 0 ])
	ones_X = np. ones (X. shape [ 0 ])
	r2 = np. outer (X2 , ones_Z ) - 2 * np. dot(X, Z.T) + np. outer (ones_X , Z2)
	return np.exp ( -0.5 / l**2 * r2)

def theta (x):
	out = 1/(1+np.exp(-x))
	return out

def cal_ll(x_train, y_train, beta_train, number, train_times, id):
    helper = np.ones((number,1))
    out_array = []
    #for index in range(0,100):
    for index in range(0,train_times):
        f = y_train*np.log(robust(theta(np.dot(beta_train[index].transpose(), x_train.transpose()))))
        s = (1 - y_train)*np.log(robust(1 - theta(np.dot(beta_train[index].transpose(), x_train.transpose()))))
        out_element = np.asscalar(np.dot((f + s),helper))
        #print(ll_element)
        out_array.append(out_element/number)
    print(out_element)
    plot_ll_color(out_array,id)
    
def cal_ll_total(x_train, y_train, beta, number):
    helper = np.ones((number,1))
    total = 0
    f = y_train*np.log(robust(theta(np.dot(beta.transpose(), x_train.transpose()))))
    s = (1 - y_train)*np.log(robust(1 - theta(np.dot(beta.transpose(), x_train.transpose()))))
    out_element = np.asscalar(np.dot((f + s),helper))
    #print(ll_element)
    total = total + out_element
    return total
    
def cal_ll_total_bay(x_train, y_train, beta, hessian_inv, number):

    ll_total_list = ll_per_point_bay(x_train, y_train, beta, hessian_inv, number)
    sum_total = sum(ll_total_list)
    return sum_total
    

def ll_per_point(x, y, beta_ll_per_point, number):
    out = []
    for index2 in range(0, number):
        ll = (math.pow(theta(np.dot(beta_ll_per_point.transpose(), x[index2].transpose())),y[index2]))*(math.pow(1 - theta(np.dot(beta_ll_per_point.transpose(), x[index2].transpose())),(1 - y[index2])))
        ll_log = np.log(ll)
        out.append(ll_log)
    return out

def ll_per_point_bay(x, y, beta_bay, hessian_inv, number):
    out = []
    for index2 in range(0, number):
        #ll = (math.pow(theta(np.dot(beta_ll_per_point.transpose(), x[index2].transpose())),y[index2]))*(math.pow(1 - theta(np.dot(beta_ll_per_point.transpose(), x[index2].transpose())),(1 - y[index2])))
        ll = (math.pow(pred_bayesian(x[index2], beta_bay, hessian_inv),y[index2]))*(math.pow(1 - pred_bayesian(x[index2], beta_bay, hessian_inv),(1 - y[index2])))
        ll_log = np.log(ll)
        out.append(ll_log)
    return out


def cal_roc(beta, x, y, num_ones, num_zeros):
    roc_array = np.array([0,0])
    for thre in range(0, 100, 1):
        t_n_roc = t_p_roc = f_n_roc = f_p_roc = 0
        for pred_roc in range(0, 300):
            p_y_roc = theta(np.dot(beta.transpose(), x[pred_roc].transpose()))
            if p_y_roc > (thre/100) and y[pred_roc] == 1:
                t_p_roc = t_p_roc + 1
            elif p_y_roc <= (thre/100) and y[pred_roc] == 1:
                f_n_roc = f_n_roc +1
            elif p_y_roc > (thre/100) and y[pred_roc] == 0:
                f_p_roc = f_p_roc + 1
            elif p_y_roc <= (thre/100) and y[pred_roc] == 0:
                t_n_roc = t_n_roc + 1
        roc_y = t_p_roc/num_ones
        roc_x = f_p_roc/num_zeros
        newrow = ([roc_x, roc_y])
        roc_array = np.vstack([roc_array, newrow])

    roc_array_x = roc_array[1:101, 0]
    roc_array_y = roc_array[1:101, 1]

    plt.axis([0, 1, 0, 1])
    plt.plot(roc_array_x, roc_array_y)
    plt.plot(roc_array_x,roc_array_x)
    plt.show()
    
    roc_array_y.sort()
    roc_array_x.sort()
    area = trapz(roc_array_y, roc_array_x) 
    print(area)
    
def cal_roc_bay(beta, x, y, num_ones, num_zeros, hessian):
    roc_array = np.array([0,0])
    for thre in range(0, 100, 1):
        t_n_roc = t_p_roc = f_n_roc = f_p_roc = 0
        for pred_roc in range(0, 300):
            #p_y_roc = theta(np.dot(beta.transpose(), x[pred_roc].transpose()))
            p_y_roc = pred_bayesian(x[pred_roc], beta, hessian)
            
            if p_y_roc > (thre/100) and y[pred_roc] == 1:
                t_p_roc = t_p_roc + 1
            elif p_y_roc <= (thre/100) and y[pred_roc] == 1:
                f_n_roc = f_n_roc +1
            elif p_y_roc > (thre/100) and y[pred_roc] == 0:
                f_p_roc = f_p_roc + 1
            elif p_y_roc <= (thre/100) and y[pred_roc] == 0:
                t_n_roc = t_n_roc + 1
        roc_y = t_p_roc/num_ones
        roc_x = f_p_roc/num_zeros
        newrow = ([roc_x, roc_y])
        roc_array = np.vstack([roc_array, newrow])

    roc_array_x = roc_array[1:101, 0]
    roc_array_y = roc_array[1:101, 1]

    plt.axis([0, 1, 0, 1])
    plt.plot(roc_array_x, roc_array_y)
    plt.plot(roc_array_x,roc_array_x)
    plt.show()
    
    roc_array_y.sort()
    roc_array_x.sort()
    area = trapz(roc_array_y, roc_array_x) 
    print(area)
    
    
def cal_confusion_matrix(beta, x, y, num_ones, num_zeros):
    t_n = t_p = f_n = f_p = 0
    for pred in range(0, 300):
        p_y = theta(np.dot(beta.transpose(), x[pred].transpose()))
        if p_y > 0.5 and y[pred] == 1:
            t_p = t_p + 1
        elif p_y <= 0.5 and y[pred] == 1:
            f_n = f_n +1
        elif p_y > 0.5 and y[pred] == 0:
            f_p = f_p + 1
        elif p_y <= 0.5 and y[pred] == 0:
            t_n = t_n + 1   

    confusion = np.array([[t_n/num_zeros, f_p/num_zeros],[f_n/num_ones, t_p/num_ones]])
    print(confusion)
    
def cal_confusion_matrix_bay(beta, x, y, num_ones, num_zeros, hessian_inv):
    t_n = t_p = f_n = f_p = 0
    for pred in range(0, 300):
        p_y = pred_bayesian(x[pred], beta, hessian_inv)
        if p_y > 0.5 and y[pred] == 1:
            t_p = t_p + 1
        elif p_y <= 0.5 and y[pred] == 1:
            f_n = f_n +1
        elif p_y > 0.5 and y[pred] == 0:
            f_p = f_p + 1
        elif p_y <= 0.5 and y[pred] == 0:
            t_n = t_n + 1   

    confusion = np.array([[t_n/num_zeros, f_p/num_zeros],[f_n/num_ones, t_p/num_ones]])
    print(confusion)
    

def cal_post_tune_neg(beta_new, sigma_sq, x, y):

    #prior_ll = (-1/(2*sigma_sq))+np.dot(beta_new.transpose(), beta_new)
    prior_ll = np.log(1/(math.sqrt(2*math.pi*sigma_sq))) + np.log(np.exp((-1/(2*sigma_sq))*np.dot(beta_new.transpose(), beta_new)))
    likelihood_ll = cal_ll_total(x, y, beta_new, 700)
    post = prior_ll + likelihood_ll
    post_neg = 0 - post
    return post_neg


def gradient(beta_new, sigma_sq, x, y):
    front = x.transpose() 
    back = y - theta(np.dot(beta_new.transpose(), x.transpose()))
    grad = (np.matmul(front,back.transpose()) - beta_new/sigma_sq)
    return (0 - grad)

    
#######################################################################################
    
######### Question c###########
X = np.loadtxt('X.txt')
y = np.loadtxt('y.txt')
plot_data(X,y)
tmp = np.c_[X,y]
np.random.shuffle(tmp)
X_new = tmp[:, [0, 1]]
y_new = tmp[:, 2]
#X_new and y_new are data after shuffle



######### Question e###########
y_training = y_new[0:700]
y_test = y_new[700:1000]

x_hat = np.c_[np.ones(1000),X_new]
x_hat_training = x_hat[0:700]
x_hat_test = x_hat[700:1000]

x_training = x_hat_training[:,1:3]
x_test = x_hat_test[:, 1:3]

beta = np.random.rand(3,1)
eta = 0.005


diff = 10
beta_array = [beta]
train_times = 0
#for p in range(0,99):
while(diff > math.pow(10, -10)):
    old_beta = beta
    front = x_hat_training.transpose() 
    back = y_training - theta(np.dot(beta.transpose(), x_hat_training.transpose()))
    beta = beta + eta*np.matmul(front,back.transpose())
    diff = np.linalg.norm(beta - old_beta)
    train_times = train_times + 1
    #print(train_times)
    beta_array.append(beta)
  
print(train_times)
# =============================================================================
# beta_array = [beta]
# for p in range(0,99):
#     front = x_hat_training.transpose() 
#     back = y_training - theta(np.dot(beta.transpose(), x_hat_training.transpose()))
#     beta = beta + eta*np.matmul(front,back.transpose())
#     beta_array.append(beta)
# =============================================================================


# =============================================================================
# helper = np.ones((700,1))
# ll_array = []
# #for q in range(0,100):
# for q in range(0,train_times):
#     first = y_training*np.log(theta(np.dot(beta_array[q].transpose(), x_hat_training.transpose())))
#     second = (1 - y_training)*np.log(1 - theta(np.dot(beta_array[q].transpose(), x_hat_training.transpose())))
#     ll_element = np.asscalar(np.dot((first + second),helper))
#     #print(ll_element)
#     ll_array.append(ll_element/700)
# plot_ll(ll_array)
# 
# helper_test = np.ones((300,1))
# ll_array_test = []
# #for r in range(0,100):
# for r in range(0,train_times):
#     first_test = y_test*np.log(theta(np.dot(beta_array[r].transpose(), x_hat_test.transpose())))
#     second_test = (1 - y_test)*np.log(1 - theta(np.dot(beta_array[r].transpose(), x_hat_test.transpose())))
#     ll_element_test = np.asscalar(np.dot((first_test + second_test),helper_test))
#     #print(ll_element)
#     ll_array_test.append(ll_element_test/300)
# plot_ll(ll_array_test)
# =============================================================================

cal_ll(x_hat_training, y_training, beta_array, 700, train_times, 0)
cal_ll(x_hat_test, y_test, beta_array, 300, train_times, 1)



def predict (X):
    X_h = np.c_[np.ones((X.shape[0],1)),X]
    #result = 1/(1+np.exp(-np.dot(beta.transpose(),X_h.transpose())))
    result = theta(np.dot(beta.transpose(), X_h.transpose()))
    return result

##
# X: 2d array with the input features
# y: 1d array with the class labels (0 or 1)
# predict : function that recives as input a feature matrix and returns a 1d
# vector with the probability of class 1.
def plot_predictive_distribution (X, y, predict ):
	xx, yy = plot_data_internal(X, y)
	ax = plt.gca()
	X_predict = np.concatenate((xx.ravel().reshape(( -1 , 1)), yy.ravel().reshape((-1 , 1))), 1)
	Z = predict( X_predict )
	Z = Z.reshape (xx.shape )
	cs2 = ax.contour(xx, yy, Z, cmap = 'RdBu', linewidths = 3)
	plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize = 14)
	plt.show()

plot_predictive_distribution (X, y, predict )


######### Question f###########

#for training#
train_ll_log = []
for train in range(0, 700):
    ll_train = (math.pow(theta(np.dot(beta.transpose(), x_hat_training[train].transpose())),y_training[train]))*(math.pow(1 - theta(np.dot(beta.transpose(), x_hat_training[train].transpose())),(1 - y_training[train])))
    ll_train_log = np.log(ll_train)
    train_ll_log.append(ll_train_log)
    
sum1_ave = sum(train_ll_log)/700
print (sum1_ave)
#for testing#
test_ll_log = []
for test in range(0, 300):
    ll_test = (math.pow(theta(np.dot(beta.transpose(), x_hat_test[test].transpose())),y_test[test]))*(math.pow(1 - theta(np.dot(beta.transpose(), x_hat_test[test].transpose())),(1 - y_test[test])))
    ll_test_log = np.log(ll_test)
    test_ll_log.append(ll_test_log)
    
sum2_ave = sum(test_ll_log)/300
print (sum2_ave)



#confusion matrix
#count number of zeros and ones in test data
y_test_list = list(y_test)
num_ones = y_test_list.count(1)
num_zeros = y_test_list.count(0)



cal_confusion_matrix(beta, x_hat_test, y_test, num_ones, num_zeros)


######### Question g###########

cal_roc(beta, x_hat_test, y_test, num_ones, num_zeros)

######### Question h###########

l = 0.1
x_train_expand = expand_inputs(l, x_training, x_training)
x_train_expand = np.c_[np.ones(700),x_train_expand]

x_test_expand = expand_inputs(l, x_test, x_training)
x_test_expand = np.c_[np.ones(300),x_test_expand]

# train beta
beta_exp = np.random.rand(701,1)
eta_exp = 0.0006


diff_rbf = 10
beta_exp_array = [beta_exp]
train_times_rbf = 0
#for p_e in range(0,99):
while(diff_rbf > 0.001):
    old_beta_exp = beta_exp
    front_exp = x_train_expand.transpose() 
    back_exp = y_training - theta(np.dot(beta_exp.transpose(), x_train_expand.transpose()))
    beta_exp = beta_exp + eta_exp*np.matmul(front_exp,back_exp.transpose())
    diff_rbf = np.linalg.norm(beta_exp - old_beta_exp)
    train_times_rbf = train_times_rbf + 1
    #print(diff_rbf)
    #print(train_times_rbf)
    beta_exp_array.append(beta_exp)


print(train_times_rbf)
cal_ll(x_train_expand, y_training, beta_exp_array, 700, train_times_rbf, 0)
cal_ll(x_test_expand, y_test, beta_exp_array, 300, train_times_rbf, 1)

ll_train_expand = ll_per_point(x_train_expand, y_training, beta_exp, 700)
ll_test_expand = ll_per_point(x_test_expand, y_test, beta_exp, 300) 

sum3_ave = sum(ll_train_expand)/700
print (sum3_ave)

sum4_ave = sum(ll_test_expand)/300
print (sum4_ave)

#plot prediction contour
def predict_rbf (X, beta):
    X_h = np.c_[np.ones((X.shape[0],1)),X]
    result = theta(np.dot(beta_exp.transpose(), X_h.transpose()))
    return result


def plot_predictive_distribution_rbf (X, y, predict, x_train, l, beta):
    xx, yy = plot_data_internal(X, y)
    ax = plt.gca()
    X_predict = np.concatenate((xx.ravel().reshape(( -1 , 1)), yy.ravel().reshape((-1 , 1))), 1)
    #X_predict = np.c_[np.ones((X_predict.shape[0],1)),X_predict]
    print(X_predict.shape)
    X_predict_rbf = expand_inputs(l, X_predict, x_train)
    Z = predict_rbf( X_predict_rbf, beta )
    Z = Z.reshape (xx.shape )
    cs2 = ax.contour(xx, yy, Z, cmap = 'RdBu', linewidths = 3)
    plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize = 14)
    plt.show()
    
    


total_x = np.vstack([x_train_expand, x_test_expand])
total_x = total_x[:,1:total_x.shape[1]]
x_test_semiexpand = x_test_expand[:,1:x_test_expand.shape[1]]
plot_predictive_distribution_rbf (X, y, predict_rbf, x_training, l, beta_exp )

#confusion matrix
cal_confusion_matrix(beta_exp, x_test_expand, y_test, num_ones, num_zeros)

#ROC
cal_roc(beta_exp, x_test_expand, y_test, num_ones, num_zeros)


#add a Gaussian prior
def cal_post_neg(beta_new):
    prior_ll = np.log(np.exp((-0.5)*np.dot(beta_new.transpose(), beta_new)))
    likelihood_ll = cal_ll_total(x_train_expand, y_training, beta_new, 700)
    post = prior_ll + likelihood_ll
    return (0 - post)

beta_map = np.zeros((701, 1))
beta_array_map = [beta_map]
eta_map = 0.005

diff_map = 10
train_times_map = 0

while(diff_map > math.pow(10, -10)):    
#while(diff_map > 0.3): 
#for p_map in range(0,99):
    old_beta_map = beta_map
    front_map = x_train_expand.transpose() 
    back_map = y_training - theta(np.dot(beta_map.transpose(), x_train_expand.transpose()))
    beta_map = beta_map + eta_map*(np.matmul(front_map,back_map.transpose()) - beta_map)
    diff_map = np.linalg.norm(beta_map - old_beta_map)
    train_times_map = train_times_map + 1
    #print(train_times_map)
    beta_array_map.append(beta_map)

cal_ll(x_train_expand, y_training, beta_array_map, 700, train_times_map, 0)
cal_ll(x_test_expand, y_test, beta_array_map, 300, train_times_map, 1)

ll_train_map = ll_per_point(x_train_expand, y_training, beta_map, 700)
ll_test_map = ll_per_point(x_test_expand, y_test, beta_map, 300) 

#confusion matrix

cal_confusion_matrix(beta_map, x_test_expand, y_test, num_ones, num_zeros)

#roc
cal_roc(beta_map, x_test_expand, y_test, num_ones, num_zeros)
    
######### Question k###########

def cal_map_hessian(sigma_sq, l):
    initial_mean_tune = np.zeros((701, 1))
    initial_cov_tune = np.identity(701)*sigma_sq
    initial_cov_inv_tune = np.linalg.inv(initial_cov_tune)
    
    x_train_expand_bay = expand_inputs(l, x_training, x_training)
    x_train_expand_bay = np.c_[np.ones(700),x_train_expand_bay]
    
    res3 = fmin_l_bfgs_b(cal_post_tune_neg, initial_mean_tune, gradient, args = (sigma_sq, x_train_expand_bay, y_training))
    beta_map3 = res3[0]

    #res_tune = minimize(cal_post_tune_neg, initial_mean_tune, args=(sigma_sq,), method='BFGS')
    #beta_map3 = res_tune.x
    # hessian
    hessian_1st_term = initial_cov_inv_tune
    
    tmp1 = theta(np.dot(beta_map3.transpose(), x_train_expand.transpose()))
    tmp2 = 1 - theta(np.dot(beta_map3.transpose(), x_train_expand.transpose()))
    tmp12 = tmp1*tmp2
 
    hessian_2nd_term = np.zeros((701,701))
    for o in range(0, 700):
        tmp3 = np.dot((np.asmatrix(x_train_expand[o,:])).transpose(), np.asmatrix(x_train_expand[o,:]))
        tmp123 = tmp12[o]*tmp3
        hessian_2nd_term = hessian_2nd_term + tmp123      
    
    hessian = hessian_1st_term + hessian_2nd_term
    hessian_inv = np.linalg.inv(hessian)
    
    return [res3, hessian]







    
# =============================================================================
# initial_mean = np.zeros((701, 1))
# initial_cov = np.identity(701)
# initial_cov_inv = np.linalg.inv(initial_cov)
# res = minimize(cal_post_neg, initial_mean, method='BFGS')
# beta_map2 = res.x
# 
# 
# hessian_1st_term = initial_cov_inv
# tmp1 = theta(np.dot(beta_map2.transpose(), x_train_expand.transpose()))
# tmp2 = 1 - theta(np.dot(beta_map2.transpose(), x_train_expand.transpose()))
# tmp12 = tmp1*tmp2
# 
# hessian_2nd_term = np.zeros((701,701))
# for o in range(0, 700):
#     tmp3 = np.dot((np.asmatrix(x_train_expand[o,:])).transpose(), np.asmatrix(x_train_expand[o,:]))
#     tmp123 = tmp12[o]*tmp3
#     hessian_2nd_term = hessian_2nd_term + tmp123      
#     
# hessian = hessian_1st_term + hessian_2nd_term
# hessian_inv = np.linalg.inv(hessian)
# 
# =============================================================================

######### Question l ###########
def kappa(sigma_sq):
    return math.pow((1 + (math.pi*sigma_sq)/8), (-0.5))
    
def pred_bayesian(x, beta, hessian_inv):
    mean_a = np.dot(beta.transpose(), x.transpose())
    sigma_a_sq = np.dot(np.asmatrix(hessian_inv), np.asmatrix(x).transpose())
    sigma_a_sq = np.dot(x, sigma_a_sq)
    tmp = kappa(sigma_a_sq)*mean_a
    tmp2 = theta(tmp)
    return tmp2


#plot prediction contour
def predict_rbf_bayesian (X, beta, hessian_inv):
    result_array = []
    X_h = np.c_[np.ones((X.shape[0],1)),X]
    for ind in range(0, X.shape[0]):
        result = pred_bayesian(X_h[ind], beta, hessian_inv)
        #result = theta(np.dot(beta_exp.transpose(), X_h.transpose()))
        result_array.append(result)
    return result_array




def plot_predictive_distribution_rbf_bay (X, y, predict, x_train, l, beta, hessian_inv):
    xx, yy = plot_data_internal(X, y)
    ax = plt.gca()
    X_predict = np.concatenate((xx.ravel().reshape(( -1 , 1)), yy.ravel().reshape((-1 , 1))), 1)
    #X_predict = np.c_[np.ones((X_predict.shape[0],1)),X_predict]
    X_predict_rbf = expand_inputs(l, X_predict, x_train)
    Z = predict_rbf_bayesian( X_predict_rbf, beta, hessian_inv)
    Z = np.array(Z)
    Z = Z.reshape (xx.shape )
    cs2 = ax.contour(xx, yy, Z, cmap = 'RdBu', linewidths = 3)
    plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize = 14)
    plt.show()
    
    
    
x_test_expand2 = expand_inputs(0.1, x_test, x_training)
x_test_expand2 = np.c_[np.ones(300),x_test_expand2]

result2 = cal_map_hessian(1, 0.1)


beta_map2= result2[0][0]
hes2 = result2[1]
hes_inv2 = np.linalg.inv(hes2)
#hes_final = result_final[1]

#plot_predictive_distribution_rbf_bay(X, y, predict_rbf_bayesian, x_training, 0.1, beta_map2, hes_inv2)
#cal_confusion_matrix_bay(beta_map2, x_test_expand2, y_test, num_ones, num_zeros, hes_inv2)
    
    
    
    
    
# contour for Bayesian classifier
print("Contour for Bayesian Classifier")
plot_predictive_distribution_rbf_bay(X, y, predict_rbf_bayesian, x_training, 0.1, beta_map2, hes_inv2)
#plot_predictive_distribution_rbf_bay(X, y, predict_rbf_bayesian, x_training, l, beta_map2, hessian_inv)
# contour for MAP method
print("Contour for MAP method")
plot_predictive_distribution_rbf(X, y, predict_rbf, x_training, 0.1, beta_map)
# Return the log_likelihoods per datapoint for Bayesian classifier and MAP method

ll_train_expand_bay = ll_per_point_bay(x_train_expand, y_training, beta_map2, hes_inv2, 700)
ll_test_expand_bay = ll_per_point_bay(x_test_expand, y_test, beta_map2, hes_inv2, 300)

ll_train_map = ll_per_point(x_train_expand, y_training, beta_map, 700)
ll_test_map = ll_per_point(x_test_expand, y_test, beta_map, 300) 
#Confusion matrix for Bayesian classifier and MAP method
print("Confusion matrix for Bayesian Classifier")
#cal_confusion_matrix_bay(beta_map2, x_test_expand, y_test, num_ones, num_zeros, hessian_inv)
cal_confusion_matrix_bay(beta_map2, x_test_expand2, y_test, num_ones, num_zeros, hes_inv2)
print("Confusion matrix for MAP method")
cal_confusion_matrix(beta_map, x_test_expand, y_test, num_ones, num_zeros)

######### Question n ###########
sigma_sq_tune = [0.45, 0.55, 0.75, 0.95, 1.15]
l_tune = [0.1, 0.3, 0.5, 0.7, 0.9]



# =============================================================================
# #add a Gaussian prior
# def cal_post_tt(beta_new):
#     prior_ll = np.log(np.exp((-0.5)*np.dot(beta_new.transpose(), beta_new)))
#     likelihood_ll = cal_ll_total(x_train_expand, y_training, beta_new, 700)
#     post = prior_ll + likelihood_ll
#     return post
# 
# def cal_post_neg(beta_new):
#     return (0 - cal_post_tt(beta_new))
# =============================================================================

    
final = np.zeros((5, 5))
for i in range(0,2):
    for j in range(0,5):
        sigma2 = sigma_sq_tune[i]
        l_rbf = l_tune[j]
        
        result = cal_map_hessian(sigma2, l_rbf)
        beta_map3 = result[0][0]
        hes = result[1]
        hes_inv = np.linalg.inv(hes)
        
        
        #x_total = np.vstack([x_training, x_test])
        #x_total_exp = expand_inputs(l_rbf, x_total, x_training)
        
        x_train_expand_bay = expand_inputs(l_rbf, x_training, x_training)
        x_train_expand_bay = np.c_[np.ones(700),x_train_expand_bay]
        
        x_test_expand_bay = expand_inputs(l_rbf, x_test, x_training)
        x_test_expand_bay = np.c_[np.ones(300),x_test_expand_bay]
        
        
        post_ll = 0 - cal_post_tune_neg(beta_map3, sigma2, x_train_expand_bay, y_training)
        post_l = np.exp(post_ll)
        #det = np.linalg.det(hes)
        #print(det)
        
        L = np.linalg.cholesky(hes)
        L_diag = np.diag(L)
        L_diag_sq = np.multiply(L_diag, L_diag)
        tp = 1
        for tp2 in range(L_diag_sq.shape[0]):
            tp = tp*L_diag_sq[tp2]
        det = tp
        print(det)
        
        likelihood = cal_ll_total(x_train_expand_bay, y_training, beta_map3, 700)
        
        #Z = lnLL - np.sum(np.power(wMAP,2)/sigma)/2-(d/2)*np.log(sigma) - np.log(det)/2
        Z_l = likelihood - np.sum(np.power(beta_map3,2)/sigma2)/2-(701/2)*np.log(sigma2) - np.log(det)/2
        
        #Z_l = post_ll + np.log(math.pow((2*math.pi), 701/2)) - np.log(math.sqrt(det))
        #exp_term_l = (-1/(2*sigma2))*np.dot(beta_map3.transpose(), beta_map3)
        #s_term = exp_term_l - Z_l
        
        #ll_train_expand_bay = ll_per_point_bay(x_train_expand_bay, y_training, beta_map3, hes_inv, 700)
        #f_term = sum(ll_train_expand_bay)
        #gaussian = multivariate_normal(mean=np.zeros(701), cov=np.identity(701)*sigma2)
        #s_term = np.log(gaussian.pdf(beta_map3))
        #t_term = (701/2)*np.log(2*math.pi)
        
        #fo_term = 0 - (0.5*(np.log(det)))
        #out = f_term + s_term + t_term + fo_term
        final[i,j] = Z_l


    
# =============================================================================
# final = []
# for i in range(0,5):
#     for j in range(0,5):
#         sigma2 = sigma_sq_tune[i]
#         l_rbf = l_tune[j]
#         
#         result = cal_map_hessian(sigma2, l_rbf)
#         beta_map4 = result[0][0]
#         hes = result[1]
#         
#         x_train_expand_bay = expand_inputs(l_rbf, x_training, x_training)
#         x_train_expand_bay = np.c_[np.ones(700),x_train_expand_bay]
#         
#         ll_train_expand_bay = ll_per_point_bay(x_train_expand_bay, y_training, beta_map4, hes, 700)
#         f_term = sum(ll_train_expand_bay)
#         s_term = (-701/2)*np.log(700)
#         
#         out = f_term + s_term
#         final.append(out)
# =============================================================================

#X_opt_train = expand_inputs(0.1, X_train, X_train)
#X_opt_test = expand_inputs(0.1, X_test, X_train)
#y_opt_train = y_train.reshape(-1)
#y_opt_test = y_test.reshape(-1)
#w0 = np.ones((X_opt_train.shape[1]+1, 1))
#beta_MAP = scipy.optimize.fmin_l_bfgs_b(func, w0, grad, args = (X_opt_train, y_opt_train, 1))[0]

#gaussian_p=multivariate_normal(mean = np.zeros(weight.shape[0]), cov = variance*np.identity(weight.shape[0]))
#second_term=np.log(gaussian_p.pdf(weight))



######### Question o ###########

x_test_expand_bay_final = expand_inputs(0.5, x_test, x_training)
x_test_expand_bay_final = np.c_[np.ones(300),x_test_expand_bay_final]

x_train_expand_bay_final = expand_inputs(0.5, x_training, x_training)
x_train_expand_bay_final = np.c_[np.ones(700),x_train_expand_bay_final]

result_final = cal_map_hessian(0.45, 0.5)


beta_map_final= result_final[0][0]
hes_final = result_final[1]
hes_inv_final = np.linalg.inv(hes_final)
#hes_final = result_final[1]

plot_predictive_distribution_rbf_bay(X, y, predict_rbf_bayesian, x_training, 0.5, beta_map_final, hes_inv_final)
cal_confusion_matrix_bay(beta_map_final, x_test_expand_bay_final, y_test, num_ones, num_zeros, hes_inv_final)

ll_train_expand_bay_final = ll_per_point_bay(x_train_expand_bay_final, y_training, beta_map_final, hes_inv_final, 700)
print(ll_train_expand_bay_final[699])
ll_test_expand_bay_final = ll_per_point_bay(x_test_expand_bay_final, y_test, beta_map_final, hes_inv_final, 300)
print(ll_test_expand_bay_final[299])