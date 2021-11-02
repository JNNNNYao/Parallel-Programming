#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
    int n = atoi(argv[1]);
    FILE *f;
    f = fopen(argv[2], "rb");
    float *data = (float *)malloc(n * sizeof(float));
    fread(data, sizeof(float), n, f);
    f = fopen(argv[3], "rb");
    float *ans = (float *)malloc(n * sizeof(float));
    fread(ans, sizeof(float), n, f);
    for(int i = 0; i < n; i++){
        if(data[i] != ans[i]){
            printf("wrong answer %s\n", argv[3]);
            printf("i: %d, data[i]: %f, ans[i]: %f\n", i, data[i], ans[i]);
            return 0;
        }
    }
    printf("pass %s\n", argv[3]);
    return 0;
}