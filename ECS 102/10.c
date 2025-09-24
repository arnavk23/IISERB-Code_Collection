#include <math.h>
#include <stdio.h>
int main() {
float p, r, t, ci;
printf("Enter principal amount: ");
scanf("%f", &p);
printf("Enter interest rate: ");
scanf("%f", &r);
printf("Enter time period in years: ");
scanf("%f", &t);
ci = p * (pow((1 + r / 100), t));
printf("Compound Interest = %f", ci);
return 0;
}
