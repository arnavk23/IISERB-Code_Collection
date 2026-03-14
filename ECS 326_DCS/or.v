module OR(input a, input b, input c, output z);

wire y;
or g1(y, a, b);
or g2(z, y, c);


endmodule 
