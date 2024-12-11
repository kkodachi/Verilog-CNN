`ifndef CNN_TOP_MODULE
`define CNN_TOP_MODULE

`include "conv2d.v"
`include "max_pool.v"
`include "fully_connected.v"

module cnn_top #(
    parameter INPUT_WIDTH = 64,
    parameter INPUT_HEIGHT = 64,
    parameter INPUT_CHANNELS = 1,
    parameter CONV_WINDOW_SIZE = 3,
    parameter CONV_NEURONS = 30,
    parameter POOL_STRIDE = 2,
    parameter FC_INPUT_SIZE = 120,
    parameter FC_OUTPUT_SIZE = 10
)(
    input wire clk,
    input wire reset,
    input wire start,
    input wire signed [15:0] input_data [0:INPUT_WIDTH-1][0:INPUT_HEIGHT-1][0:INPUT_CHANNELS-1],
    output reg signed [15:0] output_classification [0:FC_OUTPUT_SIZE-1],
    output reg cnn_done
);

    // Intermediate connection wires
    wire signed [15:0] conv_output [0:INPUT_WIDTH-CONV_WINDOW_SIZE][0:INPUT_HEIGHT-CONV_WINDOW_SIZE][0:CONV_NEURONS-1];
    wire signed [15:0] pooled_output [0:(INPUT_WIDTH-CONV_WINDOW_SIZE)/POOL_STRIDE-1]
                                     [0:(INPUT_HEIGHT-CONV_WINDOW_SIZE)/POOL_STRIDE-1]
                                     [0:CONV_NEURONS-1];
    
    // Control signals
    wire conv_done, pool_done, fc_done;
    reg conv_enable, pool_enable, fc_enable;

    // State machine
    enum {IDLE, CONV_LAYER, POOL_LAYER, FC_LAYER, DONE} state;

    // Layer modules
    conv2d #(
        .INPUT_WIDTH(INPUT_WIDTH),
        .INPUT_HEIGHT(INPUT_HEIGHT),
        .INPUT_CHANNELS(INPUT_CHANNELS),
        .WINDOW_SIZE(CONV_WINDOW_SIZE),
        .NUM_NEURONS(CONV_NEURONS)
    ) conv_layer (
        .clk(clk),
        .reset(reset),
        .enable(conv_enable),
        .input_data(input_data),
        .feature_map(conv_output),
        .conv_done(conv_done)
    );

    max_pool #(
        .INPUT_WIDTH(INPUT_WIDTH-CONV_WINDOW_SIZE+1),
        .INPUT_HEIGHT(INPUT_HEIGHT-CONV_WINDOW_SIZE+1),
        .INPUT_CHANNELS(CONV_NEURONS),
        .STRIDE(POOL_STRIDE)
    ) pool_layer (
        .clk(clk),
        .reset(reset),
        .enable(pool_enable),
        .feature_map(conv_output),
        .pooled_output(pooled_output),
        .pool_done(pool_done)
    );

    fully_connected #(
        .INPUT_SIZE(FC_INPUT_SIZE),
        .OUTPUT_SIZE(FC_OUTPUT_SIZE)
    ) fc_layer (
        .clk(clk),
        .reset(reset),
        .enable(fc_enable),
        .input_data(pooled_output[0][0]), // Flattened input
        .output_data(output_classification),
        .fc_done(fc_done)
    );

    // Top-level state machine
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            cnn_done <= 0;
            conv_enable <= 0;
            pool_enable <= 0;
            fc_enable <= 0