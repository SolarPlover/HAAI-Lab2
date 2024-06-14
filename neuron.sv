//`timescale 1ns / 1ps



module neuron #(parameter IN_SIZE=64, WIDTH = 8)(
    input clk,
    input en,
    input reset,
    input in_data[0:IN_SIZE-1],
    input weight[0:IN_SIZE-1],
    //input signed[1:0] bias,
    output signed[WIDTH-1:0] neuron_out,
    output neuron_done
    );
    
    integer addr = 0;
    reg done = 0;
    
    reg product = 0;
    reg [WIDTH-1:0] out = 0;
    
    always @(posedge clk) begin
        if(reset) begin 
            done <= 0;
            addr <= 0;
        end
        else if(en) begin
            if(addr < IN_SIZE-1) begin
                product <= in_data[addr] ^~ weight[addr]; //Calculate weighted input
                out <= out+product; //Sum each weighted input
               
            end
            if(addr == IN_SIZE-1) begin //Neuron output available
                done <= 1;
            end else begin
                addr <= addr + 1'b1;
                done <= 0;
            end
        end
    end
    
    assign neuron_out = 2 * out - IN_SIZE; 
    assign neuron_done = done;
    
endmodule
