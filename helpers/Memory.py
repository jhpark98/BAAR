import numpy as np

THRESHOLD = 0.01

class MemoryStorage:
    def __init__(self):
        self.memory = {}        # state:memory_mat
        self.state_vectors = np.array([])
    
    # FUTURE WORK: if we have too many state_vector, create a k-d tree
    def init_kdtree(self):
        pass
    
    def add_memoery(self, state_vector, memory_matrix):
        '''
        Used in the first lap to initialize the memory storage
        '''
        self.memory[state_vector] = memory_matrix
        self.state_vectors.append(state_vector)
    
    def query_memory(self, state_vector):
        # Use only the first two columns for distance calculation
        proximities = np.linalg.norm(self.state_vectors[:, :2] - state_vector[:2], axis=1)
        min_distance_index = np.argmin(proximities)
        min_distance = proximities[min_distance_index]
        
        return self.memory[self.state_vectors[min_distance_index]] if min_distance < THRESHOLD else None

    def update_memory(self, state_vector, memory_matrix):
        self.memory[state_vector] = memory_matrix

    def create_memory_matrix(self, bbox_lst):
        """
        return a memory matrix
        """
        
        # Initialize
        rows = image_size[1] // grid_size
        cols = image_size[0] // grid_size
        memory_matrix = np.zeros((rows, cols))

        for _, (x_center, y_center, width, height) in enumerate(bbox_lst):
            
            # Determine coordinate of box
            x_min = max(x_center - width / 2, 0)
            x_max = min(x_center + width / 2, image_size[0])
            y_min = max(y_center - height / 2, 0)
            y_max = min(y_center + height / 2, image_size[1])

            # Generate grid coordinates
            grid_x = np.arange(0, image_size[0], grid_size)
            grid_y = np.arange(0, image_size[1], grid_size)
            grid_x_min, grid_y_min = np.meshgrid(grid_x, grid_y)    
            grid_x_max, grid_y_max = grid_x_min + grid_size, grid_y_min + grid_size
            
            # Calculate overlap areas
            overlap_x_min = np.maximum(grid_x_min, x_min)
            overlap_y_min = np.maximum(grid_y_min, y_min)
            overlap_x_max = np.minimum(grid_x_max, x_max)
            overlap_y_max = np.minimum(grid_y_max, y_max)
            
            overlap_width = np.maximum(0, overlap_x_max - overlap_x_min)
            overlap_height = np.maximum(0, overlap_y_max - overlap_y_min)
            overlap_area = overlap_width * overlap_height
            cell_area = grid_size ** 2

            # Determine coverage
            fully_covered = (overlap_area == cell_area)
            partially_covered = (overlap_area > 0) & (overlap_area < cell_area)

            # Update memory matrix
            memory_matrix[fully_covered] = 1.0
            memory_matrix[partially_covered] = 0.6
        
        return memory_matrix

# Example usage
# image_size = (200, 100)  # Width x Height
# grid_size = 10

# bbox_lst = []
# bbox1 = (67, 74, 29, 21)  # x_center, y_center, width, height
# bbox2 = (129, 61, 53, 52)
# bbox_lst.append(bbox1)
# bbox_lst.append(bbox2)


# memory_storage = MemoryStorage()
# memory_matrix = memory_storage.create_memory_matrix(bbox_lst)