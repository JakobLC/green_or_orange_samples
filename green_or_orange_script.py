from green_or_orange_functions import *
#%% Load Image and get stats

img_path = "beads LAMP_color results.png" #if the image is in your working directory
img_path2 = "C:/Users/jakob/Desktop/Projects/Teera green or orange/\
beads LAMP_color results.png"

labels, stats, centroids, mean_hues, img = get_img_color_sample_stats(img_path2)
#%% Get a classifier
classifier = get_logistic_classifier(plot_bool=True)
#%% Classify with probabilities
probs = classify_green_orange(mean_hues,classifier=classifier)
#%% Show indices on the image
plot_img_with_indices(img,centroids)
#%% Show probabilities on the image
plot_img_with_probs(img,centroids,probs)
#%% Show individual sample bounding boxes with probabilities or indices ordered 
#from most orange to most green
show_samples_with_probs_or_idx(img,labels,probs,stats,probs_bool=True)
