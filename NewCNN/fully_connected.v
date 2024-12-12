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
    // Backpropagation
    input wire [15:0] output_error,
    input wire [15:0] learning_rate,
    output reg [15:0] input_error,
    output reg backprop_done
);

    // Fixed-point multiplication
    `define FIXED_MULT(a, b) ((a * b) >>> FIXED_POINT_BITS)

    // Memory for weights and biases
    (* ram_style = "block" *) reg [15:0] weights [0:INPUT_SIZE*OUTPUT_SIZE-1];
    (* ram_style = "block" *) reg [15:0] biases [0:OUTPUT_SIZE-1];
    
    // State machine
    reg [2:0] state;
    localparam IDLE = 3'd0;
    localparam INIT = 3'd1;
    localparam LOAD = 3'd2;
    localparam COMPUTE = 3'd3;
    localparam STORE = 3'd4;
    localparam DONE = 3'd5;

    // Computation registers
    reg [31:0] mult_result;
    reg [31:0] accumulator;
    reg [$clog2(INPUT_SIZE)-1:0] weight_idx;
    reg [15:0] weight_update;
    reg init_done;
    reg [31:0] init_counter;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= INIT;
            fc_done <= 0;
            backprop_done <= 0;
            output_valid <= 0;
            weight_idx <= 0;
            accumulator <= 0;
            init_done <= 0;
            init_counter <= 0;
            output_addr <= 0;
        end else begin
            case (state)
                INIT: begin
                    if (!init_done) begin
                        if (init_counter < INPUT_SIZE * OUTPUT_SIZE) begin
                            weights[init_counter] <= init_counter; // Initialize with sequential values for FPGA
                            init_counter <= init_counter + 1;
                        end else if (init_counter < (INPUT_SIZE * OUTPUT_SIZE + OUTPUT_SIZE)) begin
                            biases[init_counter - INPUT_SIZE * OUTPUT_SIZE] <= 0;
                            init_counter <= init_counter + 1;
                        end else begin
                            init_done <= 1;
                            state <= IDLE;
                        end
                    end
                end

                IDLE: begin
                    fc_done <= 0;
                    backprop_done <= 0;
                    output_valid <= 0;
                    if (enable && init_done) begin
                        state <= LOAD;
                        weight_idx <= 0;
                        accumulator <= {biases[output_addr], {FIXED_POINT_BITS{1'b0}}};
                    end
                end

                LOAD: begin
                    if (input_valid) begin
                        mult_result <= `FIXED_MULT(input_data, 
                                     weights[weight_idx + output_addr * INPUT_SIZE]);
                        state <= COMPUTE;
                    end
                end

                COMPUTE: begin
                    accumulator <= accumulator + mult_result;
                    
                    if (weight_idx == INPUT_SIZE-1) begin
                        state <= STORE;
                        weight_idx <= 0;
                    end else begin
                        weight_idx <= weight_idx + 1;
                        state <= LOAD;
                    end
                end

                STORE: begin
                    output_data <= accumulator[15:0];
                    output_valid <= 1;
                    
                    if (output_addr == OUTPUT_SIZE-1) begin
                        state <= DONE;
                    end else begin
                        output_addr <= output_addr + 1;
                        accumulator <= {biases[output_addr + 1], {FIXED_POINT_BITS{1'b0}}};
                        state <= LOAD;
                    end

                    // Handle backpropagation update
                    if (output_error != 0) begin
                        weight_update <= `FIXED_MULT(output_error, learning_rate);
                        weights[weight_idx + output_addr * INPUT_SIZE] <= 
                            weights[weight_idx + output_addr * INPUT_SIZE] - 
                            `FIXED_MULT(weight_update, input_data);
                        input_error <= `FIXED_MULT(output_error, 
                            weights[weight_idx + output_addr * INPUT_SIZE]);
                    end
                end

                DONE: begin
                    fc_done <= 1;
                    output_addr <= 0;
                    if (output_error != 0)
                        backprop_done <= 1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule