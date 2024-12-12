`timescale 1ns/1ps

module fully_connected #(
    parameter INPUT_SIZE = 120,
    parameter OUTPUT_SIZE = 10,
    parameter FIXED_POINT_BITS = 8
)(
    input wire clk,
    input wire reset,
    input wire enable,
    // Forward pass
    input wire [15:0] input_data,
    output reg [$clog2(INPUT_SIZE)-1:0] input_addr,
    input wire input_valid,
    output reg [15:0] output_data,
    output reg [$clog2(OUTPUT_SIZE)-1:0] output_addr,
    output reg output_valid,
    output reg fc_done,
    // Weight memory interface
    input wire [15:0] weight_data,
    output reg [$clog2(INPUT_SIZE*OUTPUT_SIZE)-1:0] weight_addr,
    input wire weight_valid,
    // Bias memory interface
    input wire [15:0] bias_data,
    output reg [$clog2(OUTPUT_SIZE)-1:0] bias_addr,
    input wire bias_valid,
    // Backpropagation
    input wire [15:0] output_error,
    input wire [15:0] learning_rate,
    output reg [15:0] input_error,
    output reg backprop_done
);

    // State machine
    reg [3:0] state;
    localparam IDLE = 4'd0;
    localparam LOAD_BIAS = 4'd1;
    localparam LOAD_WEIGHT = 4'd2;
    localparam COMPUTE = 4'd3;
    localparam ACCUMULATE = 4'd4;
    localparam STORE = 4'd5;
    localparam BACKPROP = 4'd6;
    localparam UPDATE_WEIGHTS = 4'd7;
    localparam DONE = 4'd8;

    // Computing registers
    reg [31:0] mult_result;
    reg [31:0] accumulator;
    reg [$clog2(INPUT_SIZE)-1:0] input_idx;
    reg [$clog2(OUTPUT_SIZE)-1:0] output_idx;
    reg [15:0] weight_update;

    // Fixed-point multiplication
    function [15:0] fixed_mult;
        input [15:0] a;
        input [15:0] b;
        reg [31:0] temp;
    begin
        temp = a * b;
        fixed_mult = temp[23:8]; // 16-bit result with 8 fractional bits
    end
    endfunction

    // Main process
    always @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            fc_done <= 0;
            backprop_done <= 0;
            output_valid <= 0;
            input_addr <= 0;
            weight_addr <= 0;
            bias_addr <= 0;
            output_addr <= 0;
            input_idx <= 0;
            output_idx <= 0;
            accumulator <= 0;
        end else begin
            case (state)
                IDLE: begin
                    if (enable) begin
                        state <= LOAD_BIAS;
                        output_valid <= 0;
                        fc_done <= 0;
                        backprop_done <= 0;
                    end
                end

                LOAD_BIAS: begin
                    if (bias_valid) begin
                        accumulator <= {bias_data, {FIXED_POINT_BITS{1'b0}}};
                        state <= LOAD_WEIGHT;
                        input_idx <= 0;
                    end
                    bias_addr <= output_idx;
                end

                LOAD_WEIGHT: begin
                    if (weight_valid && input_valid) begin
                        weight_addr <= input_idx + output_idx * INPUT_SIZE;
                        input_addr <= input_idx;
                        state <= COMPUTE;
                    end
                end

                COMPUTE: begin
                    mult_result <= fixed_mult(input_data, weight_data);
                    state <= ACCUMULATE;
                end

                ACCUMULATE: begin
                    accumulator <= accumulator + mult_result;
                    
                    if (input_idx == INPUT_SIZE-1) begin
                        state <= STORE;
                    end else begin
                        input_idx <= input_idx + 1;
                        state <= LOAD_WEIGHT;
                    end
                end

                STORE: begin
                    output_data <= accumulator[23:8];
                    output_valid <= 1;
                    output_addr <= output_idx;

                    if (output_idx == OUTPUT_SIZE-1) begin
                        state <= output_error ? BACKPROP : DONE;
                    end else begin
                        output_idx <= output_idx + 1;
                        state <= LOAD_BIAS;
                    end
                end

                BACKPROP: begin
                    if (weight_valid && input_valid) begin
                        // Calculate weight updates
                        weight_update <= fixed_mult(output_error, learning_rate);
                        input_error <= fixed_mult(output_error, weight_data);
                        
                        if (input_idx == INPUT_SIZE-1 && output_idx == OUTPUT_SIZE-1) begin
                            state <= DONE;
                        end else if (input_idx == INPUT_SIZE-1) begin
                            input_idx <= 0;
                            output_idx <= output_idx + 1;
                        end else begin
                            input_idx <= input_idx + 1;
                        end
                    end
                end

                DONE: begin
                    fc_done <= 1;
                    if (output_error) backprop_done <= 1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule