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
    input wire [15:0] input_data,
    output reg [$clog2(INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS)-1:0] input_addr,
    input wire input_valid,
    output reg [15:0] feature_map,
    output reg [$clog2((INPUT_WIDTH - WINDOW_SIZE + 1) * (INPUT_HEIGHT - WINDOW_SIZE + 1) * NUM_NEURONS)-1:0] output_addr,
    output reg output_valid,
    output reg conv_done,
    input wire [15:0] output_error,
    input wire [15:0] learning_rate,
    output reg backprop_done
);

    // Memory for weights
    reg [15:0] kernel [0:INPUT_CHANNELS * WINDOW_SIZE * WINDOW_SIZE * NUM_NEURONS - 1];
    reg [$clog2(INPUT_CHANNELS * WINDOW_SIZE * WINDOW_SIZE * NUM_NEURONS)-1:0] kernel_addr;

    // State machine
    reg [2:0] state;
    localparam IDLE = 3'd0, FORWARD = 3'd1, BACKPROP = 3'd2, DONE = 3'd3;

    reg [31:0] accumulator;

    initial begin
        kernel_addr = 0;
        for (int i = 0; i < INPUT_CHANNELS * WINDOW_SIZE * WINDOW_SIZE * NUM_NEURONS; i = i + 1)
            kernel[i] = $random % 256; // Random initial weights
    end

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            conv_done <= 0;
            backprop_done <= 0;
        end else begin
            case (state)
                IDLE: begin
                    if (enable) state <= FORWARD;
                end
                FORWARD: begin
                    // Convolution Logic
                    if (input_valid) begin
                        accumulator = accumulator + input_data * kernel[kernel_addr];
                        kernel_addr = kernel_addr + 1;
                        if (kernel_addr == INPUT_CHANNELS * WINDOW_SIZE * WINDOW_SIZE * NUM_NEURONS - 1) begin
                            feature_map = accumulator[15:0];
                            output_valid = 1;
                            state <= BACKPROP;
                        end
                    end
                end
                BACKPROP: begin
                    // Weight update logic
                    kernel[kernel_addr] = kernel[kernel_addr] - (output_error * learning_rate) >> FIXED_POINT_BITS;
                    backprop_done = 1;
                    state <= DONE;
                end
                DONE: begin
                    conv_done = 1;
                    state <= IDLE;
                end
            endcase
        end
    end
endmodule
