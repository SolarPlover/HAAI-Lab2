//`timescale 1ns / 1ps



module neuron_input_layer #(parameter IN_SIZE=784, WIDTH = 8)(
    input clk,
    input en,
    input reset,
    input [WIDTH-1:0] in_data[0:IN_SIZE-1],
    input weight[0:IN_SIZE-1],
    //input signed[1:0] bias,
    output signed[4*WIDTH-1:0] neuron_out,
    output neuron_done
    );
    
    integer addr = 0;
    reg done = 1'b0;
    
    reg signed [4*WIDTH-1:0] product = 0;
    reg signed [4*WIDTH-1:0] out = 0;
    reg signed [4*WIDTH-1:0] twos_complement = 0;
    
    always @(posedge clk) begin
        if(reset) begin 
            done <= 1'b0;
            addr <= 0;
        end
        else if(en) begin
            if(addr < IN_SIZE-1) begin
                twos_complement <= ~{24'b0, in_data[addr]} + 1'b1;
                product <= weight[addr] ? {24'b0, in_data[addr]} : twos_complement; //Calculate weighted input
                out <= out+product; //Sum each weighted input
               
            end
            if(addr == IN_SIZE-1) begin //Neuron output available
                done <= 1'b1;
            end else begin
                addr <= addr + 1'b1;
                done <= 1'b0;
            end
        end
    end
    
    assign neuron_out = out; //Add bias
    assign neuron_done = done;
    
endmodule