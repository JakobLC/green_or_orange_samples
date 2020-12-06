import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.linear_model import LogisticRegression

def get_img_color_sample_stats(img_path, min_size=300, max_size=3000, 
                           saturation_thresh=25, bbox_fill_min=0.6):
    """
    Function description:
        Finds useful stats from an image of N corona orange/green colored 
        samples. The function maps the RGB image to HSV and uses the saturation
        to find the ROIs of the colored samples. The mean hue over each of
        these ROIs is returned as this is used for later classification.
    ----------
    Inputs
    ----------
    img_path : str
        String of the path where the given image of samples of orange/green is 
        saved.
    min_size=300 : float, int
        Minimum connected component size (in pixels) the algorithm considers
        a sample.
    max_size=3000 : float, int
        Maximum connected component size (in pixels) the algorithm considers
        a sample.
    saturation_thresh=25 : float, int, [0,255]
        Thresholding value used to binarize the saturation channel from HSV
        before doing connected component analysis.
    bbox_fill_min=0.6 : float, [0.0,1.0]
        Minimum ratio of connected component area divided by bounding box area
        that the algorithm considers as a sample.
    ----------
    Outputs
    ----------
    labels : 2d np.array, int32
        An image containing the values zero on background and value i=1..N
        in ROI number i. Same size as img.
    stats : 2d np.array, int32
        shape(N, 5) Array of stats where column 0,1 is coordinates for the top left 
        corner of the ROI. Column 2,3 is the width of the bounding box of the
        ROI. Column 4 is the area in pixels of the ROI.
    centroids : 2d np.array, float64
        shape(N, 2) Array of geometrical centers for the ROIs.
    mean_hues : 1d np.array, float64
        shape(N,) Array of average hue values of the ROIs in the HSV colormap.
    img : 3d np.array, uint8
        RGB image of the samples from the given image path.
    ----------
    Written by Jakob Loenborg Christensen Dec. 2020
    """
    img = Image.open(img_path)
    img = np.array(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat = hsv[:,:,1]
    hue = hsv[:,:,0]
    ret, th = cv2.threshold(sat, saturation_thresh, 255, cv2.THRESH_BINARY)
	
    ret, labels1, stats, centroids = cv2.connectedComponentsWithStats(th,connectivity=8)
	
    use = (stats[:,4] <= max_size) & (stats[:,4] >= min_size)
    use = (use) & ((stats[:,4]/stats[:,[2,3]].prod(1))>=bbox_fill_min)
    
    labels2 = np.zeros_like(labels1)
    mean_hues = []
    for i,idx in enumerate(np.nonzero(use)[0]):
        labels2[labels1==idx] = i 
        mean_hues.append(hue[labels1==idx].mean())
    return labels2, stats[use,:], centroids[use,:], np.array(mean_hues), img

def plot_img_with_indices(img,centroids):
    """
    Function description:
        Shows an image with the points given by centroids. Each point is shown
        with their index on the image.
    ----------
    Inputs
    ----------
    img : 3d np.array, uint8
        RGB image of the samples from the given image path.
    centroids : 2d np.array, float64
        shape(N, 2) Array of geometrical centers for the ROIs.
    ----------
    Written by Jakob Loenborg Christensen Dec. 2020
    """
    fig, ax = plt.subplots()
    (x,y) = (centroids[:,0],centroids[:,1])
    plt.imshow(img)
    ax.scatter(x,y,color=(0,0,0),alpha=0.5,marker='x')
    for i in range(x.size):
        ax.annotate(i, (x[i],y[i]))

def plot_img_with_probs(img,centroids,probs):
    """
    Function description:
        Shows an image with the points given by centroids. Each point is shown
        with their probability of being a green sample, given by probs.
    ----------
    Inputs
    ----------
    img : 3d np.array, uint8
        RGB image of the samples from the given image path.
    centroids : 2d np.array, float64
        shape(N, 2) Array of geometrical centers for the ROIs.
    probs : 1d np.array, float64
        Probabilities of samples being green.
    ----------
    Written by Jakob Loenborg Christensen Dec. 2020
    """
    fig, ax = plt.subplots()
    (x,y) = (centroids[:,0],centroids[:,1])
    plt.imshow(img)
    ax.scatter(x,y,color=(0,0,0),alpha=0.5,marker='x')
    for i, txt in enumerate(probs):
        ax.annotate(txt.round(2), (x[i],y[i]))   
   
def get_logistic_classifier(plot_bool=False,
                            mean_hues=np.array([31.2, 15.4, 28.6, 17.6, 16.9]),
                            ground_truth=np.array([1,0,1,0,0])):
    """
    Function description:
        Returns a logistic classifier. If no data to train this is supplied
        then it returns the classifier trained on the 5 controls in the image
        "beads LAMP_color results.png"
    ----------
    Inputs
    ----------
    plot_bool=True : Boolean
        Boolean variable deciding if the classifier and training points be 
        plotted.
    mean_hues : 1d np.array, float64
        shape(N,) Array of average hue values of the ROIs in the HSV colormap.
    ground_truth : 1d np.array, float, int
        Ground truth where 1=green sample and 0=orange sample. Ordering and
        shape has to match mean_hues.
    ----------
    Outputs
    ----------
    clf : sklearn LogisticRegression object
        Classifier that can be used with the function "classify_green_orange"
    ----------
    Written by Jakob Loenborg Christensen Dec. 2020
    """
    clf = LogisticRegression(random_state=0).fit(mean_hues.reshape(-1, 1), 
                                                 ground_truth)
    if plot_bool:
        t = np.linspace(mean_hues.min(),mean_hues.max(),100)
        plt.scatter(mean_hues,ground_truth,color=[0.8,0.2,0.2])
        ty = classify_green_orange(t,classifier=clf)
        plt.plot(t,ty)
        plt.legend(["Logistic classifier","Training data"])
        plt.xlabel("Hue")
        plt.ylabel("Probability of green sample")
    return clf

def classify_green_orange(mean_hues,classifier=get_logistic_classifier()):
    """
    Function description:
        Classifies Mean hue values as either green or orange with a probability
        of being green.
    ----------
    Inputs
    ----------
    mean_hues : 1d np.array, float64
        shape(N,) Array of average hue values of the ROIs in the HSV colormap.
    classifier : sklearn LogisticRegression object
        Classifier that was returned from the function 
        "get_logistic_classifier"
    ----------
    Outputs
    ----------
    probs : 1d np.array, float64
        Probability of each mean hue value being a green sample. Same size as
        mean_hues.
    ----------
    Written by Jakob Loenborg Christensen Dec. 2020
    """
    probs = classifier.predict_proba(mean_hues.reshape(-1, 1))[:,1]
    return probs

def show_samples_with_probs_or_idx(img,labels,probs,stats,probs_bool=True):
    """
    Function description:
        Plots all the samples given by labels, and stats from the image.
        Can show either probabilities or 
    ----------
    Inputs
    ----------
    img : 3d np.array, uint8
        RGB image of the samples from the given image path.
    labels : 2d np.array
        An image containing the values zero on background and value i=1..N
        in ROI number i. Same size as img.
    probs : 1d np.array, float64
        Probability of each mean hue value being a green sample. Same size as
        mean_hues.
    probs_bool=True : Boolean
        Boolean variable deciding if the titles of the subplots should display
        probabilities of being a green sample (True) or the index of the
        samples (False).
    ----------
    Written by Jakob Loenborg Christensen Dec. 2020
    """
    idx = np.argsort(probs)
    n = np.round(1.5*np.sqrt(idx.size)).astype(int)
    m = np.ceil(idx.size/n).astype(int)
    fig, axs = plt.subplots(nrows=m,ncols=n)
    k=0
    for i in range(m):
        for j in range(n):
            if k<idx.size:
                axs[i,j].imshow(img[stats[idx[k],1]:(stats[idx[k],1]+stats[idx[k],3]),
                                stats[idx[k],0]:(stats[idx[k],0]+stats[idx[k],2])])
                if probs_bool:
                    axs[i,j].set_title("p="+str((probs[idx[k]]).round(2)))
                else:
                    axs[i,j].set_title("i="+str(idx[k]))
                axs[i,j].axis('off')
            else:
                fig.delaxes(axs[i,j])
            k+=1

    
    