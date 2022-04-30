import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.image as img
import imageio
import copy



def create_random_means_array(k, n):
    """ This method creates a random 2D array of size k x n.  The numbers in the array are random numbers between 0 and 1.
    The purpose is to initialize randomly the locations of our k centroids in n-dimensional space."""
    # We initialize our 2D array to all zeros.
    means_array = np.zeros((k, n))

    for k_index in range(0, k):
        # Create initial sub-array.
        sub_array = np.zeros(n)
        for n_index in range(0, n):
            # Get a random number.
            rand = np.random.random()
            sub_array[n_index] = rand
        # Now sub_array is n units long and each is a random number between 0 and 1.
        means_array[k_index] = sub_array
    return means_array


def distance_3D(x1, y1, z1, x2, y2, z2):
    """ This method determines the Euclidean distance in RGB (3 Dimensional) space."""
    dist = np.square(x2 - x1) + np.square(y2 - y1) + np.square(z2 - z1)
    dist = np.sqrt(dist)
    return dist


def k_means(im, k, max_iterations):
    """ This method performs the kmeans algorithm.  First, some small preperation steps are made, and then algorithm repeats until convergence.
    Each pixel is assigned a mean, and then the means are updated.  IF the algorithm does not converge by the time max_iterations is reached,
    then the algorithm stops."""
    points = np.reshape(im, (im.shape[0] * im.shape[1], im.shape[2]))
    m, n = points.shape
    means_array = create_random_means_array(k, n)


    # We count up the number of iterations we have done until we reach the max_iterations.
    iteration_count = 1

    # We initialize an old_index array, which will be used later to check convergence.
    old_index_array = np.zeros(2)


    while(iteration_count <= max_iterations):
        # Initialize new_index_array to all zeros.
        new_index_array = np.zeros(m)

        print("New iteration of k-means algorithm...")

        for j in range(len(points)):
            # Initialize a min_dist value.  At the end the distance from current point in 3D to the nearest centroid.
            min_dist = 1000

            for k_index in range(0, k):

                x1 = points[j, 0]
                y1 = points[j, 1]
                z1 = points[j, 2]
                x2 = means_array[k_index, 0]
                y2 = means_array[k_index, 1]
                z2 = means_array[k_index, 2]

                if(distance_3D(x1, y1, z1, x2, y2, z2) < min_dist):
                    min_dist = distance_3D(x1, y1, z1, x2, y2, z2)
                    new_index_array[j] = k_index

        # At this point, we check to see if centroid's constituents have changed.  If not, we have converged and are done.
        if len(old_index_array) != 2:
            # Then we are not on the first iteration, and the comparison makes sense.
            if np.array_equal(old_index_array, new_index_array):
                # If we reach here, we can just return the items.
                return means_array, new_index_array
        print("Have Not Converged Yet!")

        # At this point, index_array now holds an index representing the closest cluster to each pixel RGB value.
        # Now we adjust the location of the means based on the average values of its constituents.
        for k_index in range(0, k):

            sumx = 0
            sumy = 0
            sumz = 0
            count = 0

            for j in range(len(points)):

                if(new_index_array[j] == k_index):
                    sumx += points[j, 0]
                    sumy += points[j, 1]
                    sumz += points[j, 2]
                    count += 1

            # If count == 0, then the current cluster has 0 constituents :(, but we need to make count = 1, so the next part doesnt blow up.
            if(count == 0):
                count = 1

            means_array[k_index, 0] = float(sumx / count)
            means_array[k_index, 1] = float(sumy / count)
            means_array[k_index, 2] = float(sumz / count)

        # All of the above is a single iteration, so we decrement and go again.
        iteration_count = iteration_count + 1
        # The old index becomes a copy of the new index, so we can check convergence on the next iteration.
        old_index_array = copy.deepcopy(new_index_array)

    # The result of the above method is an altered means_array, so we return that and the index_array.
    return means_array, new_index_array


def compress_image(means_array, index_array, im):
    """ This method assigns every pixel the value of its "parent centroid".  The resulting compressed image is formed by reshaping,
    and the image is displayed and saved. """

    centroid = np.array(means_array)
    recovered = centroid[index_array.astype(int), :]

    # We turn the 2D array that we have been working with back to a 3D array.
    recovered = np.reshape(recovered, (im.shape[0], im.shape[1],
                                                     im.shape[2]))

    # We display the compressed image.
    plt.imshow(recovered)
    plt.show()

    # We save the new compressed image.  The below will have to be adjusted.
    imageio.imwrite('C:\\Users\\ryanc\\Dropbox\\CS 6375 - Machine Learning\\homework\\hw3\\RESULT.jpg', recovered)

    # Now for the sake of this homework, we need to find the size of the resulting compressed file (so we can find compression ratio)
    compressed_size = os.path.getsize('C:\\Users\\ryanc\\Dropbox\\CS 6375 - Machine Learning\\homework\\hw3\\RESULT.jpg')

    return compressed_size


if __name__ == '__main__':

    # We set the default parameters.
    k = 10
    max_iterations = 10
    image_location = 'C:\\Users\\ryanc\\Dropbox\\CS 6375 - Machine Learning\\homework\\hw3\\Penguins.jpg'

    # Now we adjust the parameters based on the inputs to the command line.
    if len(sys.argv) == 4:
        print("You provided the correct number of parameters.  Congrats!")
        k = int(sys.argv[1])
        max_iterations = int(sys.argv[2])
        image_location = sys.argv[3]
    else:
        print("You did not provide the correct number of parameters.  Using default selections.")


    # First we load the image.
    im = imageio.imread(image_location)

    # Now we make all of the numbers between 0 and 1.
    im = im / 255

    # Then we run the kmeans algorithm which will update the means_array and return an index_array (assigning all pixels to a cluster).
    new_means_array, index_array = k_means(im, k, max_iterations)

    # Then we reconstruct the image using the new_means_array and index_array.  The image should open.
    compressed_size = compress_image(new_means_array, index_array, im)

    old_size = os.path.getsize(image_location)

    compression_ratio = old_size / compressed_size

    print("Compression Ratio: ", compression_ratio)
