import numpy as np

class gd_group19_2d:
    
    # Initialize gd_pv_2d class attributes, such as the loss function and the its partial derivatives
    def __init__(self, fn_loss, fn_grad_x1, fn_grad_x2): 
        self.fn_loss = fn_loss
        self.fn_grad_x1 = fn_grad_x1
        self.fn_grad_x2 = fn_grad_x2
        
    def find_min(self, x1_init, x2_init, n_iter, eta, tol):
        x1 = x1_init #starting point
        x2 = x2_init #starting point
                
        loss_path = [] #loss_path records the path of the loss function
        x1_path = [] #initialize x1_path as an empty matrix
        x2_path = []
        
        x1_path.append(x1) #put the starting point into the path
        x2_path.append(x2)
        
        loss_this = self.fn_loss(x1,x2) #evaluate loss function at starting point (x1,x2)
        loss_path.append(loss_this) #save this in loss_path matrix
        g1 = self.fn_grad_x1(x1,x2) #find partial derivarive wrt to x1(gradient) at starting point
        g2 = self.fn_grad_x2(x1,x2) #same for x2

        for i in range(n_iter): # loop until we iterate n_iter times or until both g1,g2 < tol
            if (abs(g1) < tol and abs(g2) < tol) or  np.isnan(g1) or  np.isnan(g2):
                break
            g1 = self.fn_grad_x1(x1,x2) # find g1
            g2 = self.fn_grad_x2(x1,x2)
            x1 += -eta * g1 # find new x1 which is x1 <- x1 - eta * g1
            x2 += -eta * g2
            x1_path.append(x1) #save new x1 to list x1_path
            x2_path.append(x2)
            loss_this = self.fn_loss(x1,x2) #calculate loss function and save it loss_path
            loss_path.append(loss_this)
        
        if np.isnan(g1) or np.isnan(g2):
            print('Exploded')
        elif np.abs(g1) > tol or  np.abs(g2) > tol:
            print('Did not converge')
        else:
            print('Converged in {} steps.  Loss fn {} achieved by (x1,x2) = ({},{})'.format(i+1, np.round(loss_this, 12),np.round(x1, 7), np.round(x2, 7)))
            #print('The minimum of the loss function is {}'.format())
            #print('The value of x1 which generates the minimum is {}'.format())
            #print('The value of x2 which generates the minimum is {}'.format())
            #print('The number of steps is {}'.format(np.round(i+1,7)))    
        
        #self.loss_path = np.array(loss_path)
        #self.x_path = np.array(x_path)
        self.loss_path = loss_path
        self.x1_path = x1_path
        self.x2_path = x2_path
        self.loss_fn_min = loss_this
        self.x1_at_min = x1
        self.x2_at_min = x2
        self.x1_g = g1
        self.x2_g = g2
        self.num_steps = i+1
        
    def momentum(self, x1_init, x2_init, n_iter, eta, tol, alpha):
        x1 = x1_init
        x2 = x2_init
        
        loss_path = []
        x1_path = []
        x2_path = []
        
        x1_path.append(x1)
        x2_path.append(x2)
        loss_this = self.fn_loss(x1,x2)
        loss_path.append(loss_this)
        g1 = self.fn_grad_x1(x1,x2)
        g2 = self.fn_grad_x2(x1,x2)
        
        nu1 = 0
        nu2 = 0

        for i in range(n_iter):
            g1 = self.fn_grad_x1(x1,x2)
            g2 =  self.fn_grad_x2(x1,x2)

            if (abs(g1) < tol and abs(g2) < tol) or  np.isnan(g1) or  np.isnan(g2):
                break
                
            nu1 = alpha * nu1 + eta * g1
            nu2 = alpha * nu2 + eta * g2
            x1 += -nu1
            x2 += -nu2
            x1_path.append(x1)
            x2_path.append(x2)
            loss_this = self.fn_loss(x1,x2)
            loss_path.append(loss_this)

        if  np.isnan(g1) or np.isnan(g2):
            print('Exploded')
        elif  np.abs(g1) > tol or  np.abs(g2) > tol:
            print('Did not converge')
        else:
            print('Converged in {} steps.  Loss fn {} achieved by (x1,x2) = ({},{})'.format(i+1, np.round(loss_this, 12),np.round(x1, 7), np.round(x2, 7)))
       
        self.loss_path = loss_path
        self.x1_path = x1_path
        self.x2_path = x2_path
        self.loss_fn_min = loss_this
        self.x1_at_min = x1
        self.x2_at_min = x2
        self.x1_g = g1
        self.x2_g = g2
        self.num_steps = i+1
        
 
    def rmsprop(self, x1_init, x2_init, n_iter, eta, tol, beta):
        
        x1 = x1_init
        x2 = x2_init
        
        loss_path = []
        x1_path = []
        x2_path = []
        
        x1_path.append(x1)
        x2_path.append(x2)
        
        loss_this = self.fn_loss(x1,x2)
        loss_path.append(loss_this)
        g1 = self.fn_grad_x1(x1,x2)
        g2 = self.fn_grad_x2(x1,x2)
              
        g1_mean_sq = 0
        g2_mean_sq = 0
        eps = 1e-8
        
        for i in range(n_iter):       
            if (abs(g1) < tol and abs(g2) < tol) or  np.isnan(g1) or  np.isnan(g2):
                break
            
            g1 = self.fn_grad_x1(x1,x2)
            g2 =  self.fn_grad_x2(x1,x2)
                        
            g1_mean_sq = beta * g1_mean_sq + (1 - beta) * g1 ** 2
            g2_mean_sq = beta * g2_mean_sq + (1 - beta) * g2 ** 2
            x1 -= eta * g1 / (np.sqrt(g1_mean_sq) + eps)
            x2 -= eta * g2 / (np.sqrt(g2_mean_sq) + eps)
            
            x1_path.append(x1)
            x2_path.append(x2)
            loss_this = self.fn_loss(x1,x2)
            loss_path.append(loss_this)

        if  np.isnan(g1) or np.isnan(g2):
            print('Exploded')
        elif  np.abs(g1) > tol or  np.abs(g2) > tol:
            print('Did not converge')
        else:
            print('Converged in {} steps.  Loss fn {} achieved by (x1,x2) = ({},{})'.format(i+1, np.round(loss_this, 12),np.round(x1, 7), np.round(x2, 7)))
       
        self.loss_path = loss_path
        self.x1_path = x1_path
        self.x2_path = x2_path
        self.loss_fn_min = loss_this
        self.x1_at_min = x1
        self.x2_at_min = x2
        self.x1_g = g1
        self.x2_g = g2
        self.num_steps = i+1
        
    def adam(self, x1_init, x2_init, n_iter, eta, tol, alpha, beta):        
        x1 = x1_init
        x2 = x2_init
        
        loss_path = []
        x1_path = []
        x2_path = []
        
        x1_path.append(x1)
        x2_path.append(x2)
        
        loss_this = self.fn_loss(x1,x2)
        loss_path.append(loss_this)
        g1 = self.fn_grad_x1(x1,x2)
        g2 = self.fn_grad_x2(x1,x2)
              
        g1_mean_sq = 0
        g2_mean_sq = 0
        m1 = 0
        m2 = 0
        eps = 1e-8
  
        for i in range(n_iter):       
            if (abs(g1) < tol and abs(g2) < tol) or  np.isnan(g1) or  np.isnan(g2):
                break
            
            g1 = self.fn_grad_x1(x1,x2)
            g2 =  self.fn_grad_x2(x1,x2)
            
            m1 = alpha * m1 + (1-alpha) * g1 
            m2 = alpha * m2 + (1-alpha) * g2
            g1_mean_sq = beta * g1_mean_sq + (1 - beta) * g1 ** 2
            g2_mean_sq = beta * g2_mean_sq + (1 - beta) * g2 ** 2
            
            #Corrected estimators to remove bias
            m1_hat = m1 / (1 - np.power(alpha, i+1))
            m2_hat = m2 / (1 - np.power(alpha, i+1))
            v1_hat = g1_mean_sq / (1 - np.power(beta, i+1))
            v2_hat = g2_mean_sq / (1 - np.power(beta, i+1))
             
            x1 -= eta * m1_hat / (np.sqrt(v1_hat) + eps)
            x2 -= eta * m2_hat / (np.sqrt(v2_hat) + eps)
            
            x1_path.append(x1)
            x2_path.append(x2)
            loss_this = self.fn_loss(x1,x2)
            loss_path.append(loss_this)
  
        if  np.isnan(g1) or np.isnan(g2):
            print('Exploded')
        elif  np.abs(g1) > tol or  np.abs(g2) > tol:
            print('Did not converge')
        else:
            print('Converged in {} steps.  Loss fn {} achieved by (x1,x2) = ({},{})'.format(i+1, np.round(loss_this, 12),np.round(x1, 7), np.round(x2, 7)))
       
        self.loss_path = loss_path
        self.x1_path = x1_path
        self.x2_path = x2_path
        self.loss_fn_min = loss_this
        self.x1_at_min = x1
        self.x2_at_min = x2
        self.x1_g = g1
        self.x2_g = g2
        self.num_steps = i+1