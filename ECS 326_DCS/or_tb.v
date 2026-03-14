`include "or.sv"

module or_tb();

	reg A, B, C;
	wire out;

	OR G2(A, B, C, out);
	
	initial begin
	
	$dumpfile("or_tb.vcd");
	$dumpvars(0, or_tb);
	
	A = 0; B = 0; C = 0;
        #10 A = 1; B = 0; C = 0;
        #10 A = 0; B = 1; C = 0;
        #10 A = 1; B = 1; C = 0;
        #10 A = 0; B = 0; C = 1;
        #10 A = 1; B = 0; C = 1;
        #10 A = 0; B = 1; C = 1;
        #10 A = 1; B = 1; C = 1;
        
	
	end
endmodule
