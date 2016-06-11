#include <stdio.h>

int main()
{
    float y = 0;
    float vy = 10000000000000;
    float dt = 0.1;
    float L = 1e-4;
    y = y + vy*dt;
    printf("%f %f %f\n", y, floor(y/L)*L, y - floor(y/L)*L);
    y = y - floor(y/L)*L;
}
