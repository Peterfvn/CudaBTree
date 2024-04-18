// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

struct Node {
    int val;
    struct Node* next;
};
Node* head;

__global__ void linkedList(Node* node) {
    int tid = threadIdx.x;
    
    int count = 0;
    while(node->next && count < tid) {
        count++;
        node = node->next;
    }
    printf("%d ", node->val);
}

void insertNode(Node* node, int val) {
    while(node->next) {
        node = node->next;
    }
    Node* current = (Node*)malloc(sizeof(Node));
    node->next = current;
    current->val = val;
    current->next = NULL;
}

void unifiedInsert(Node* node, int val) {
    while(node->next) {
        node = node->next;
    }
    Node* current;
    cudaMallocManaged(&current, sizeof(Node));
    node->next = current;
    current->val = val;
    current->next = NULL;
}

void printList(Node* node) {
    while(node) {
        printf("%d ", node->val);
        node = node->next;
    }
    printf("\n");
}

int main() {

    // head = (struct Node*)malloc(sizeof(Node));
    cudaMallocManaged(&head, sizeof(Node));
    head->val = 1;
    head->next = NULL;

    // insertNode(head, 5);
    // insertNode(head, 10);
    // insertNode(head, 15);
    
    unifiedInsert(head, 5);

    linkedList<<<1, 2>>>(head);
    cudaDeviceSynchronize();

    // printList(head);


    return 1;
}