#include <iostream>
using namespace std;
#define L 1e-4
#define N_grid 16
#define dx (L/float(N_grid))

void test(float x)
{
    float y = x - floor(x/L)*L;
    cout << x << " " << y << endl;
}
int main()
{
    cout << "L:" << L << endl;
    test(-3*L);
    test(-2*L);
    test(-L);
    test(-0.5*L);
    test(0);
    test(L/2.0);
    test(L);
    test(L*1.5);
    test(3*L);
}
