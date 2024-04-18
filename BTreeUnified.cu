// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <stdlib.h>
#include <math.h>

#define MAX 3
#define MIN 2

struct BTreeNode {
  int val[MAX + 1], count = 0;
  struct BTreeNode *link[MAX + 1];
};

struct BTreeNode *root;

// Create a node
struct BTreeNode *createNode(int val, struct BTreeNode *child) {
  struct BTreeNode *newNode;
//   newNode = (struct BTreeNode *)malloc(sizeof(struct BTreeNode));
  cudaMallocManaged(&newNode, sizeof(struct BTreeNode));
  newNode->val[1] = val;
  newNode->count = 1;
  newNode->link[0] = root;
  newNode->link[1] = child;
  return newNode;
}

// Insert node
void insertNode(int val, int pos, struct BTreeNode *node,
        struct BTreeNode *child) {
  int j = node->count;
  while (j > pos) {
    node->val[j + 1] = node->val[j];
    node->link[j + 1] = node->link[j];
    j--;
  }
  node->val[j + 1] = val;
  node->link[j + 1] = child;
  node->count++;
}

// Split node
void splitNode(int val, int *pval, int pos, struct BTreeNode *node,
         struct BTreeNode *child, struct BTreeNode **newNode) {
  int median, j;

  if (pos > MIN)
    median = MIN + 1;
  else
    median = MIN;

//   *newNode = (struct BTreeNode *)malloc(sizeof(struct BTreeNode));
    cudaMallocManaged(newNode, sizeof(struct BTreeNode));
  j = median + 1;
  while (j <= MAX) {
    (*newNode)->val[j - median] = node->val[j];
    (*newNode)->link[j - median] = node->link[j];
    j++;
  }
  node->count = median;
  (*newNode)->count = MAX - median;

  if (pos <= MIN) {
    insertNode(val, pos, node, child);
  } else {
    insertNode(val, pos - median, *newNode, child);
  }
  *pval = node->val[node->count];
  (*newNode)->link[0] = node->link[node->count];
  node->count--;
}

// Set the value
int setValue(int val, int *pval,
           struct BTreeNode *node, struct BTreeNode **child) {
  int pos;
  if (!node) {
    *pval = val;
    *child = NULL;
    return 1;
  }

  if (val < node->val[1]) {
    pos = 0;
  } else {
    for (pos = node->count;
       (val < node->val[pos] && pos > 1); pos--)
      ;
    if (val == node->val[pos]) {
      printf("Duplicates are not permitted\n");
      return 0;
    }
  }
  if (setValue(val, pval, node->link[pos], child)) {
    if (node->count < MAX) {
      insertNode(*pval, pos, node, *child);
    } else {
      splitNode(*pval, pval, pos, node, *child, child);
      return 1;
    }
  }
  return 0;
}

// Insert the value
void insert(int val) {
  int flag, i;
  struct BTreeNode *child;

  flag = setValue(val, &i, root, &child);
  if (flag)
    root = createNode(i, child);
}

// Search node
void search(int val, int *pos, struct BTreeNode *myNode) {
  if (!myNode) {
    return;
  }

  if (val < myNode->val[1]) {
    *pos = 0;
  } else {
    for (*pos = myNode->count;
       (val < myNode->val[*pos] && *pos > 1); (*pos)--)
      ;
    if (val == myNode->val[*pos]) {
      printf("%d is found\n", val);
      return;
    }
  }
  search(val, pos, myNode->link[*pos]);

  return;
}

// Traverse then nodes
void traversal(struct BTreeNode *myNode) {
  int i;
  if (myNode) {
    for (i = 0; i < myNode->count; i++) {
      traversal(myNode->link[i]);
      printf("%d ", myNode->val[i + 1]);
    }
    traversal(myNode->link[i]);
  }
}

// Calculations to determine how big the array should be

int height = 5;
int sizeOfArr = (pow((MAX + 1), height+1)-1)/((MAX+1)-1);
struct BTreeNode** treeArray = (struct BTreeNode**)malloc(sizeof(struct BTreeNode*) * (sizeOfArr));

// Create array for tree. Does not work currently
void makeArray(struct BTreeNode* myNode) {
    int front = 0, rear = 0, index = 0;
    struct BTreeNode** queue = (struct BTreeNode**) malloc(sizeof(struct BTreeNode*) * 1000);
    struct BTreeNode* current;

    queue[rear++] = myNode;
    while(front < rear) {
        current = queue[front++];
        int i;
        if(current) {
            treeArray[index++] = current;
            for(i = 0; i < current->count; i++) {
                queue[rear++] = current->link[i];
            }
            queue[rear++] = current->link[i];
        } else {
            treeArray[index++] = NULL;
        }

    }

}

void levelOrderTraversal(struct BTreeNode* myNode) {
    int front = 0, rear = 0;
    struct BTreeNode** queue = (struct BTreeNode**)malloc(sizeof(struct BTreeNode*) * 1000);
    struct BTreeNode* current;

    queue[rear++] = myNode;
    while(front < rear) {
        current = queue[front++];
        int i;
        if(current) {
            for(i = 0; i < current->count; i++) {
                printf("%d ", current->val[i+1]);
                queue[rear++] = current->link[i];
            }
            queue[rear++] = current->link[i];
        }
    }
}

// Stack related functions for non-recursive traversal
struct sNode {
    struct BTreeNode* t;
    struct sNode* next;
};

void push(struct sNode** top, struct BTreeNode* t) {
    // struct sNode* newNode = (struct sNode*)malloc(sizeof(struct sNode));
    struct sNode* newNode;
    cudaMallocManaged(&newNode, sizeof(struct sNode));

    newNode->t = t;
    newNode->next = *top;
    *top = newNode;
}

struct BTreeNode* pop(struct sNode** top_ref) {
    struct BTreeNode* res;
    struct sNode* top;

    top = *top_ref;
    res = top->t;
    *top_ref = top->next;
    free(top);
    return res;

}

bool isEmpty(struct sNode* top) {
    return (top==NULL) ? 1 : 0;
}

void iterativeInOrderTraversal(BTreeNode* root, int depth) {
    int count = 0;
    struct BTreeNode* current = root;
    struct sNode* stack = NULL;
    push(&stack, current);
    bool done = 0;

    while(!isEmpty(stack) && count < depth) {
        BTreeNode* node = pop(&stack);
        for(int i = 1; i <= node->count; i++) {
            printf("%d ", node->val[i]);
        }
        if(node != NULL) {
            for(int i = 0; i <= MAX; i++) {
                if(node->link[i]) {
                    push(&stack, node->link[i]);
                }
            }
        }
    }
}

__device__ void pushGPU(struct sNode** top_ref, struct BTreeNode* t) {
    struct sNode* newNode;
    newNode = (sNode*)malloc(sizeof(struct sNode));

    newNode->t = t;
    newNode->next = *top_ref;
    *top_ref = newNode;
}

__device__ BTreeNode* popGPU(struct sNode** top_ref) {
    struct BTreeNode* res;
    struct sNode* top;

    top = *top_ref;
    res = top->t;
    *top_ref = top->next;
    free(top);
    return res;
}

__device__ bool isEmptyGPU(struct sNode* top) {
    return (top == NULL) ? 1 : 0;
}

__device__ void GPU_DFS(BTreeNode* root, int tid) {
    struct BTreeNode* current = root;
    struct sNode* stack = NULL;
    pushGPU(&stack, current);
    int depth = 0;

    while(!isEmptyGPU(stack) && depth <= tid) {

        current = popGPU(&stack);
        for(int i = 1; i<= current->count; i++) {
            printf("%d ", current->val[i]);
        }
        printf("\n");

        if(current != NULL) {
            for(int i = 0; i <= MAX; i++) {
                if(current->link[i]) {
                    pushGPU(&stack, current->link[i]);
                }
            }
        }
        depth++;
    }
}

__global__ void printTree(BTreeNode* root) {
    int tid = threadIdx.x;
    GPU_DFS(root, tid);
}

__global__ void bruteForceTree(BTreeNode* root) {
    int tid = threadIdx.x;
    printf("%d, %d ", root->val[1], root->val[2]);
    printf("\n%d, %d | %d, %d | %d\n", root->link[0]->val[1], root->link[0]->val[2], root->link[1]->val[1], root->link[1]->val[2], root->link[2]->val[1]);
}

int main() {
  int ch;

  insert(8);
  insert(9);
  insert(10);
  insert(11);
  insert(15);
  insert(16);
  insert(17);
//   insert(18);
//   insert(20);
//   insert(23);
//   insert(24);
//   insert(25);
//   insert(26);

    // iterativeInOrderTraversal(root, 5);
    // printf("\n");
    printTree<<<1, 2>>>(root);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

//   bruteForceTree<<<1, 1>>>(root);
//   cudaDeviceSynchronize();


}
