class Node:
    def __init__(self, value: float = 0, parent: Node = None, leftChild: Node = None, sibling: Node = None):
        self.value = value
        self.parent = parent
        self.leftChild = leftChild
        self.sibling = sibling


