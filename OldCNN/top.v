`timescale 1ns/1ps

module cnn #(
    parameter INPUT_WIDTH = 64,
    parameter INPUT_HEIGHT = 64,
    parameter INPUT_CHANNELS = 1,
    parameter CONV_WINDOW_SIZE = 3,
    parameter NUM_NEURONS = 30,
    parameter POOL_STRIDE = 2,
    parameter FIXED_POINT_FRACTIONAL_BITS = 8
)(
    input wire clk,
    input wire reset,
    input wire enable,
    
    // Input data interface
    input wire signed [15:0] input_data,
    output wire [$clog2(INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS)-1:0] input_addr,
    input wire input_valid,
    
    // Output data interface
    output wire signed [15:0] output_data,
    output wire [$clog2((INPUT_WIDTH/POOL_STRIDE)*(INPUT_HEIGHT/POOL_STRIDE)*NUM_NEURONS)-1:0] output_addr,
    output wire output_valid,
    
    // Control signals
    output wire cnn_done
);
    // Intermediate wires between layers
    wire signed [15:0] conv_output_data;
    wire [$clog2((INPUT_WIDTH-CONV_WINDOW_SIZE+1)*(INPUT_HEIGHT-CONV_WINDOW_SIZE+1)*NUM_NEURONS)-1:0] conv_output_addr;
    wire conv_output_valid;
    wire conv_done;

    wire signed [15:0] pool_input_data;
    wire [$clog2((INPUT_WIDTH-CONV_WINDOW_SIZE+1)*(INPUT_HEIGHT-CONV_WINDOW_SIZE+1)*NUM_NEURONS)-1:0] pool_input_addr;
    wire pool_input_valid;

    // Control signals between layers
    wire conv_enable, pool_enable;
    wire pool_done;

    // State machine for layer sequencing
    localparam IDLE = 2'd0;
    localparam CONV = 2'd1;
    localparam POOL = 2'd2;
    localparam COMPLETE = 2'd3;

    reg [1:0] current_state;
    reg layer_enable;

    // Layer instantiations
    conv2d #(
        .INPUT_WIDTH(INPUT_WIDTH),
        .INPUT_HEIGHT(INPUT_HEIGHT),
        .INPUT_CHANNELS(INPUT_CHANNELS),
        .WINDOW_SIZE(CONV_WINDOW_SIZE),
        .NUM_NEURONS(NUM_NEURONS),
        .FIXED_POINT_FRACTIONAL_BITS(FIXED_POINT_FRACTIONAL_BITS)
    ) conv_layer (
        .clk(clk),
        .reset(reset),
        .enable(conv_enable),
        .input_data(input_data),
        .input_addr(input_addr),
        .input_valid(input_valid),
        .feature_map(conv_output_data),
        .output_addr(conv_output_addr),
        .output_valid(conv_output_valid),
        .conv_done(conv_done)
    );

    max_pool #(
        .INPUT_WIDTH(INPUT_WIDTH-CONV_WINDOW_SIZE+1),
        .INPUT_HEIGHT(INPUT_HEIGHT-CONV_WINDOW_SIZE+1),
        .INPUT_CHANNELS(NUM_NEURONS),
        .STRIDE(POOL_STRIDE)
    ) pool_layer (
        .clk(clk),
        .reset(reset),
        .enable(pool_enable),
        .input_data(conv_output_data),
        .input_addr(pool_input_addr),
        .input_valid(conv_output_valid),
        .pooled_output(output_data),
        .output_addr(output_addr),
        .output_valid(output_valid),
        .pool_done(pool_done)
    );

    // Layer sequencing state machine
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            current_state <= IDLE;
            layer_enable <= 0;
        end else begin
            case (current_state)
                IDLE: begin
                    if (enable) begin
                        current_state <= CONV;
                        layer_enable <= 1;
                    end
                end

                CONV: begin
                    if (conv_done) begin
                        current_state <= POOL;
                        layer_enable <= 1;
                    end
                end

                POOL: begin
                    if (pool_done) begin
                        current_state <= COMPLETE;
                        layer_enable <= 0;
                    end
                end

                COMPLETE: begin
                    current_state <= IDLE;
                end
            endcase
        end
    end

    // Layer enable signal generation
    assign conv_enable = (current_state == CONV) & layer_enable;
    assign pool_enable = (current_state == POOL) & layer_enable;
    assign cnn_done = (current_state == COMPLETE);
endmodule