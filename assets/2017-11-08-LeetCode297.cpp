/** 
    @file 2017-11-08-LeetCode297.cpp
    Leet Code 297: Serialize/Deserialize Binary Tree
    Purpose: This is my best solution for Leet Code Problem 297. The task is to create a Codec class
             is capable of representing a binary tree of int's as a single string, and the also
             being able to convert from this string representation to a binary tree of int's.

             My solution doesn't store empty nodes (i.e. leaves) and also uses special separating characters
             (instead of commas) to designate which children a node has.

             At the time this solution was submitted, it runs faster than 
             89.06% of all successful cpp submissions for this problem.

    @author Matthew McGonagle
*/

#include<iostream>
#include<string>
#include<vector>
#include<queue>
#include<sstream>
#include<iostream>

using namespace std;

/**
    Struct TreeNode is for holding node information. The format is specified by the Leet Code
    problem
*/
struct TreeNode {
    int val; // Value held at node
    TreeNode *left, *right; // Pointers to children.

    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

/**
    Class Codec is for serializing and deserializing the binary tree.

    The format of the string is that the nodes are entered in a breadth first traversal manner.
    Instead of using a single separator (such as a comma), we will use a suffix after each node
    value based on the type of children the node has. The letters are:
        'T' the node has two non-empty children.
        'L' the node has only a left non-empty child.
        'R' the node has only a right non-empty child.
        'E' the node has no children. Both of its children are null (i.e. empty). 
    The end of the tree is represented using the ']' character.
*/

class Codec {
    
    public:
       
        /** 
            Turns a tree into its string representation.
            
            @param root This is a pointer to the root node of the tree.
            @return The string representation of the tree.
        */ 
        string serialize(TreeNode* root); // Specified by Leet Code.

        /** 
            Turns a string into the binary tree it represents.
        
            @param data The string data is the representation of the tree.
            @return A pointer to the root node of the tree.
        */
        TreeNode* deserialize(string data); // Specified by Leet Code.
    
};

// Functions for testing our solution.

/** 
    Builds a tree from a vector array of ints. However, a particular value of integer represents a null node (i.e. empty node). Note, this tree is NOT a binary search tree, and it is allowed to contain repeat values. Nodes are simply constructed in a breadth first manner using the order of the array.
    @param nums The array of numbers to use to construct the tree. The nodes are entered in a breadth first manner.
    @param nullVal Integers with the value nullVal will not be stored as full nodes. Instead, these indicate empty nodes, i.e. the end of a particular branch in the tree. 
*/
TreeNode* buildTree(vector<int> nums, int nullVal);

/**
    A simple print of the tree. Prints each level on a separate line.
    @param root This variable points to the root node of the tree.
*/
void printTree(TreeNode* root);

/**
    Deletes a tree from memory.
    @param root This variable points to the root node of the tree.
*/ 
void deleteTree(TreeNode* root);


/**
    The main executable function. This is used to test our solution before submitting to Leet Code.
*/
int main() {

    int nullVal = -100,
        numsArray[] = {1, 2, 10, 3, -100, 5, 6, 3, 1, -100, 2, 4, -100, 6};
    vector<int> nums = vector<int>(numsArray, numsArray + sizeof(numsArray) / sizeof(int));
    TreeNode *root, *deserialized;
    Codec codec;
    string serialization;

    // Construct the tree. Print it to see that it works.
    root = buildTree(nums, nullVal);
    printTree(root);

    // Now test the codec serialization and deserialization.

    cout << "Now let's print the serialization of the tree." << endl;
    serialization = codec.serialize(root);
    cout << serialization << endl;

    cout << "Now let's test the deserialization of this string." << endl;
    deserialized = codec.deserialize(serialization); 
    printTree(deserialized);

    // Clean up.

    deleteTree(root);
    deleteTree(deserialized);
    return 0;
}

////////// Function definitions

string Codec::serialize( TreeNode* root ) {

    ostringstream ss;
    queue<TreeNode*> toProcess; // holds node needed to be processed.
    TreeNode *current;

    // Case of empty tree.

    if(root == NULL)
        return string("]");

    // We will traverse the nodes in a breadth first manner.

    toProcess.push(root);

    while(!toProcess.empty()) {

        current = toProcess.front();
        toProcess.pop();

        ss << current -> val;
        
        // Determine the suffix for the node value based on the node's children. 

        if (current -> left != NULL && current -> right != NULL){ 
            ss << 'T';
            toProcess.push(current -> left);
            toProcess.push(current -> right);
        }
        else if (current -> left != NULL) { 
            ss << 'L';
            toProcess.push(current -> left);
        }
        else if (current -> right != NULL) {
            ss << 'R';
            toProcess.push(current -> right);
        }
        else 
            ss << 'E';
        
    }

    // Mark the end of the tree.

    ss << ']';

    return ss.str();   
}

TreeNode* Codec::deserialize(string data) {

    TreeNode *root, *newNode, **newParent;
    queue<TreeNode**> parents; // Holds pointer to parents child pointer to assign to current node.
                               // Using double pointers to allow putting left and right children
                               // in the same queue. Keeps everything much more simple.
    stringstream ss(data); // For parsing string.
    int val;
    char nodeType; // Suffix type of node.

    // Case of Empty String.
    if(data.size() == 0)
        return NULL;

    // Check to see if string represents empty tree.
    if(ss.peek() == ']')
        return NULL;

    // Set up the initial root.

    ss >> val;
    ss >> nodeType;
    root = new TreeNode(val);
    newNode = root;

    if (nodeType == 'T') {
        parents.push(&(newNode -> left));
        parents.push(&(newNode -> right));
    }
    else if (nodeType == 'L') 
        parents.push(&(newNode -> left));
    else if (nodeType == 'R') 
        parents.push(&(newNode -> right));

    // Now construct the rest of the tree while we haven't reached the end of the tree.

    while (ss.peek() != ']') {

        ss >> val;
        ss >> nodeType;
        
        newNode= new TreeNode(val);
        
        // Front of parents queue (which must be non-empty as this node must have a parent) holds
        // a pointer to which child (left or right) this node is relative to its parent.
        if(!parents.empty()) {
            newParent = parents.front();
            parents.pop();
            *newParent = newNode;
        }
       
        // Use the suffix of this node to figure out which child pointer(s) to load into parents queue. 

        if (nodeType == 'T') {
            parents.push(&(newNode -> left));
            parents.push(&(newNode -> right));
        }
        else if (nodeType == 'L') 
            parents.push(&(newNode -> left));
        else if (nodeType == 'R') 
            parents.push(&(newNode -> right));

    } 

    return root;
}

// Build a tree from an array of numbers. Use nullVal to indicate empty nodes.

TreeNode* buildTree(vector<int> nums, int nullVal) {

    TreeNode *root, *current;
    queue<TreeNode*> leafnodes; // A queue of the current leaf nodes to add children too.
    vector<int>::iterator iT = nums.begin() + 1;

    if (nums.size() == 0)
        return NULL;

    root = new TreeNode(nums[0]);
    leafnodes.push(root);

    // For each loop, we try to add both a left and right child. So, we try to add two nodes
    // for each pass. Except the case of nullVal.

    while(iT != nums.end()) {

        current = leafnodes.front();
        leafnodes.pop();

        // Look at adding left child.

        if(*iT != nullVal) {
            current -> left = new TreeNode(*iT);
            leafnodes.push(current -> left);
        }
        iT++;

        // Look at adding right child.

        if(iT != nums.end()) {
       
            if(*iT != nullVal) { 
                current -> right = new TreeNode(*iT);
                leafnodes.push(current -> right);
            }
            iT++;
        } 
    }

    return root;
    
}

void printTree(TreeNode* root) {

    // Use two queues. At any given time one queue holds nodes to print on current
    // level while other holds values on next level.

    queue<TreeNode*> toPrintA, toPrintB;
    queue<TreeNode*> *toPrintNow, *toPrintNext, *swapper;
    TreeNode* current;

    if(root == NULL) {
        cout << "NULL";
        return;
    }

    toPrintNow = &toPrintA;
    toPrintNext = &toPrintB;    
    toPrintNow -> push(root);

    while(!toPrintNow -> empty()) {

        current = toPrintNow -> front();
        toPrintNow -> pop();

        if(current == NULL)
            cout << "NULL, ";
        else {
            cout << current -> val << ", ";
            toPrintNext -> push(current -> left);
            toPrintNext -> push(current -> right);
        }

        if(toPrintNow -> empty()) {
            // Swap pointer values.
            swapper = toPrintNow;
            toPrintNow = toPrintNext;
            toPrintNext = swapper;
            cout << endl; 
        }

    } 
    
}

// Traverse the tree in a breadth first manner using a queue inorder to free the nodes from memory.

void deleteTree(TreeNode* root) {

    queue<TreeNode*> toDelete;
    TreeNode *current;

    if(root == NULL)
        return;
    
    toDelete.push(root);
    while(!toDelete.empty()) {
        
        current = toDelete.front();
        toDelete.pop();

        if ( current -> left != NULL )
            toDelete.push( current -> left );
        if ( current -> right != NULL )
            toDelete.push( current -> right );
        
        delete current;
    }
}
