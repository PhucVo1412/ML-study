

class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, label=None):
        self.feature_index = feature_index        
        self.threshold = threshold   
        self.left = left              
        self.right = right            
        self.label = label             # Leaf node label prediction

class BinaryDecisionTree:
    def __init__(self, max_depth=3):
        self.root = None
        self.max_depth = max_depth

    def gini_index(self, groups, classes ):
        total_samples = sum([len(group) for group in groups])
        gini = 0.0
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            score = 0.0
        for class_val in classes:
            proportion = [row[-1] for row in group].count(class_val) /size
            score += proportion ** 2
        gini += (1.0- score) * (size / total_samples)
        return gini

    def split_dataset(self, dataset, index, threshold):
        if type(threshold) == str:
            left = [row for row in dataset if row[index] == threshold]
            right = [row for row in dataset if row[index] != threshold]

        left = [row for row in dataset if row[index] < threshold]
        right = [row for row in dataset if row[index] >= threshold]

        return left, right

    def best_split(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        best_index, best_threshold, best_gini, best_groups = None, None, float('inf'), None

        for index in range(len(dataset[0]) - 1):
            for row in dataset:
                groups = self.split_dataset(dataset, index, row[index])
                gini = self.gini_index(groups, class_values)
                if gini < best_gini:
                    best_index, best_threshold, best_gini, best_groups = index, row[index], gini, groups
        return best_index, best_threshold, best_groups
    
    def build_tree(self, dataset, depth=0):
        class_values = [row[-1] for row in dataset]

        if len(set(class_values)) == 1 or depth >= self.max_depth:
            return TreeNode(label=max(set(class_values), key=class_values.count))
        
        feature_index, thereshold, (left, right) = self.best_split(dataset)

        if not left or not right:
            return TreeNode(label=max(set(class_values), key=class_values.count))
        
        left_node = self.build_tree(left, depth + 1)
        right_node = self.build_tree(right, depth + 1)

        return TreeNode(feature_index=feature_index, threshold=thereshold, left=left_node, right=right_node)
    
    def fit(self, dataset):
        self.root = self.build_tree(dataset)

    def print_tree(self, node=None, depth=0):

        if node is None:
            node = self.root

        if node.label is not None:
            print(f"{' '  * depth} [Leaf] Label: {node.label}")
        else:
            if isinstance(node.threshold, str):
                print(f"{' ' * depth} [Node] Feature {node.feature_index} =={node.threshold}")
            else:
                print(f"{' ' * depth} [Node] Feature {node.feature_index} <={node.threshold}")
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)



dataset =[
    [2.8,'Yes'],
    [1.2,'No'],
    [3.6,'Yes'],
    [4.5,'No'],
    [5.1,'Yes']
]

tree = BinaryDecisionTree(max_depth=3)
tree.fit(dataset)

tree.print_tree()


dataset2 = [
    ['Yes','Yes','No','No'],
    ['Yes','No','No','No'],
    ['No','Yes','No','Yes'],
    ['No','No','Yes','Yes'],
    ['Yes','Yes','No','Yes'],
    ['No','Yes','Yes','No'],
    ['Yes','No','Yes','Yes'],
    ['No','No','No','No']
]

tree.fit(dataset2)
tree.print_tree()
