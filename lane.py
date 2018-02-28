import numpy as np
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = []  
        #calculated curvatures for the last n iterations
        self.curvatures = [] 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        
    def add_fit(self, fit, inds):
        # add a found fit to the line, up to n
        if fit is not None:
            if self.best_fit is not None:
                # if we have a best fit, see how this new fit compares
                self.diffs = abs(fit-self.best_fit)
            
            if (self.diffs[0] > 0.1 or self.diffs[1] > 1.5 or self.diffs[2] > 150.) and len(self.current_fit) > 0:
                self.detected = False
            else:
                self.detected = True
                self.current_fit.append(fit)
                if len(self.current_fit) > 5:
                    # throw out old fits, keep newest n
                    self.current_fit = self.current_fit[len(self.current_fit)-5:]
                self.best_fit = np.average(self.current_fit, axis=0)
        # or remove one from the history, if not found
        else:
            self.detected = False    
            if len(self.current_fit) > 0:
                # throw out oldest fit
                self.current_fit = self.current_fit[:len(self.current_fit)-1]
                if len(self.current_fit) > 0:
                    # if there are still any fits in the queue, best_fit is their average
                    self.best_fit = np.average(self.current_fit, axis=0)
                
    def add_curvature(self, curvature):
        #append curvatures list
        if len(self.curvatures) > 60:
            self.curvatures = self.curvatures[:len(self.current_fit)-1]
        self.curvatures.append(curvature);
            
    def get_line_curvature(self):
        #returns average line curvature over last n iterations
        return np.average(self.curvatures)