`timescale 1ns/1ps

module conv2d #(
    parameter INPUT_WIDTH = 64,
    parameter INPUT_HEIGHT = 64,
    parameter INPUT_CHANNELS = 1,
    parameter WINDOW_SIZE = 3,
    parameter NUM_NEURONS = 30,
    parameter FIXED_POINT_BITS = 8
)(
    input wire clk,
    input wire reset,
    input wire enable,
    // Forward pass
    input wire [15:0] input_data,
    output reg [$clog2(INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS)-1:0] input_addr,
    input wire input_valid,
    output reg [15:0] feature_map,
    output reg [$clog2((INPUT_WIDTH-WINDOW_SIZE+1)*(INPUT_HEIGHT-WINDOW_SIZE+1)*NUM_NEURONS)-1:0] output_addr,
    output reg output_valid,
    output reg conv_done,
    // Backpropagation
    input wire [15:0] output_error,
    input wire [15:0] learning_rate,
    output reg backprop_done
);

    // Fixed-point multiplication
    `define FIXED_MULT(a, b) ((a * b) >>> FIXED_POINT_BITS)

    // Memory for weights
    reg [15:0] kernel [0:INPUT_CHANNELS*WINDOW_SIZE*WINDOW_SIZE*NUM_NEURONS-1];
    reg [$clog2(INPUT_CHANNELS*WINDOW_SIZE*WINDOW_SIZE*NUM_NEURONS)-1:0] kernel_addr;

    // State machine
    reg [2:0] state;
    localparam IDLE = 3'd0;
    localparam INIT_WEIGHTS = 3'd1;
    localparam COMPUTE = 3'd2;
    localparam STORE = 3'd3;
    localparam BACKPROP = 3'd4;
    localparam DONE = 3'd5;

    // Computation registers
    reg [31:0] conv_accumulator;
    reg [$clog2(WINDOW_SIZE)-1:0] win_i, win_j;
    reg [$clog2(INPUT_CHANNELS)-1:0] channel;
    reg [15:0] weight_update;

    // Weight initialization
    integer i;
    initial begin
        for (i = 0; i < INPUT_CHANNELS*WINDOW_SIZE*WINDOW_SIZE*NUM_NEURONS; i = i + 1)
            kernel[i] = $random;
    end

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            conv_done <= 0;
            backprop_done <= 0;
            output_valid <= 0;
        end else begin
            case (state)
                IDLE: begin
                    if (enable) begin
                        state <= COMPUTE;
                        conv_accumulator <= 0;
                        win_i <= 0;
                        win_j <= 0;
                        channel <= 0;
                    end
                end

                COMPUTE: begin
                    if (input_valid) begin
                        conv_accumulator <= conv_accumulator + 
                            `FIXED_MULT(input_data, kernel[kernel_addr]);
                        
                        if (win_j == WINDOW_SIZE-1) begin
                            win_j <= 0;
                            if (win_i == WINDOW_SIZE-1) begin
                                win_i <= 0;
                                state <= STORE;
                            end else
                                win_i <= win_i + 1;
                        end else
                            win_j <= win_j + 1;
                    end
                end

                STORE: begin
                    feature_map <= conv_accumulator[15:0];
                    output_valid <= 1;
                    if (output_error != 0) // If error present, do backprop
                        state <= BACKPROP;
                    else
                        state <= DONE;
                end

                BACKPROP: begin
                    // Update weights based on error
                    weight_update <= `FIXED_MULT(output_error, learning_rate);
                    kernel[kernel_addr] <= kernel[kernel_addr] - weight_update;
                    
                    if (kernel_addr == INPUT_CHANNELS*WINDOW_SIZE*WINDOW_SIZE*NUM_NEURONS-1)
                        state <= DONE;
                    else
                        kernel_addr <= kernel_addr + 1;
                end

                DONE: begin
                    conv_done <= 1;
                    if (output_error != 0)
                        backprop_done <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule