import cv2

# Path to your image
image_path = 'test_export/0_jpg.rf.02b6f90631ccfb1252425f7d49568522.jpg'

# Path to your coordinates file
coordinates_file = 'test_export/0_jpg.rf.02b6f90631ccfb1252425f7d49568522.txt'

# Read the coordinates from the file
with open(coordinates_file, 'r') as f:
    lines = f.readlines()

# Load your image
image = cv2.imread(image_path)

# Counter for naming cropped images
counter = 1

# Process each line of coordinates
for line in lines:
    # Split the line into individual values
    values = line.strip().split()

    # Convert values to floats (assuming they are in string format)
    values = list(map(float, values))

    # Extract coordinates
    x2 = int(values[1] * image.shape[1])  # Calculate x1 in image coordinates
    y2 = int(values[2] * image.shape[0])  # Calculate y1 in image coordinates
    x1 = int(values[3] * image.shape[1])  # Calculate x2 in image coordinates
    y1 = int(values[4] * image.shape[0])  # Calculate y2 in image coordinates
    
    print("Coordinates:", x1, x2, y1, y2)

    # Crop the region from the image
    cropped_region = image[y1:y2, x1:x2]

    if cropped_region.size > 0:
        cv2.imwrite('cropped_image_{}.jpg'.format(counter), cropped_region)
        counter += 1
    else:
        print(f"Error: Cropped region {counter} is empty or invalid.")
