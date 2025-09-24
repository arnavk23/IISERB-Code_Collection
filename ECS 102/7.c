#include <stdio.h>

int main(){
int num1,num2;
printf("input number 1 :");
scanf("%d",&num1);
printf("input number 2 :");
scanf("%d",&num2);
if (num1>0 && num2>0){
    printf("the product is positive");
}
if (num1<0 && num2<0){
    printf("the product is positive");
}
else {
    printf("the product is negative");
}
return 0;
}
