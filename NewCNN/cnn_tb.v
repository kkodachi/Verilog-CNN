`timescale 1ns/1ps

module cnn_tb();
    // Parameters
    parameter INPUT_WIDTH = 64;
    parameter INPUT_HEIGHT = 64;
    parameter INPUT_CHANNELS = 1;
    parameter WINDOW_SIZE = 3;
    parameter NUM_NEURONS = 30;
    parameter POOL_STRIDE = 2;
    parameter FC_OUTPUT_SIZE = 10;
    parameter FIXED_POINT_BITS = 8;

    // Clock and reset
    reg clk;
    reg reset;
    
    // Control signals
    reg conv_enable, pool_enable, fc_enable;
    wire conv_done, pool_done, fc_done;

    // Data signals
    reg [15:0] input_data;
    wire [15:0] conv_output;
    wire [15:0] pool_output;
    wire [15:0] fc_output;
    reg [15:0] learning_rate;
    
    // Memory interface signals
    wire conv_output_valid, pool_output_valid, fc_output_valid;
    reg input_valid;
    wire [$clog2(INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS)-1:0] conv_input_addr;
    wire [$clog2(FC_OUTPUT_SIZE)-1:0] fc_output_addr;
    
    // Test data storage
    reg [15:0] test_data [0:INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS-1];
    reg [15:0] expected_output [0:FC_OUTPUT_SIZE-1];

    // Instantiate modules
    conv2d #(
        .INPUT_WIDTH(INPUT_WIDTH),
        .INPUT_HEIGHT(INPUT_HEIGHT),
        .INPUT_CHANNELS(INPUT_CHANNELS),
        .WINDOW_SIZE(WINDOW_SIZE),
        .NUM_NEURONS(NUM_NEURONS)
    ) conv_layer (
        .clk(clk),
        .reset(reset),
        .enable(conv_enable),
        .input_data(input_data),
        .input_valid(input_valid),
        .feature_map(conv_output),
        .output_valid(conv_output_valid),
        .conv_done(conv_done),
        .input_addr(conv_input_addr)
    );

    max_pool #(
        .INPUT_WIDTH(INPUT_WIDTH-WINDOW_SIZE+1),
        .INPUT_HEIGHT(INPUT_HEIGHT-WINDOW_SIZE+1),
        .INPUT_CHANNELS(NUM_NEURONS),
        .STRIDE(POOL_STRIDE)
    ) pool_layer (
        .clk(clk),
        .reset(reset),
        .enable(pool_enable),
        .input_data(conv_output),
        .input_valid(conv_output_valid),
        .pooled_output(pool_output),
        .output_valid(pool_output_valid),
        .pool_done(pool_done)
    );

    fully_connected #(
        .INPUT_SIZE((INPUT_WIDTH-WINDOW_SIZE+1)*(INPUT_HEIGHT-WINDOW_SIZE+1)*NUM_NEURONS/(POOL_STRIDE*POOL_STRIDE)),
        .OUTPUT_SIZE(FC_OUTPUT_SIZE)
    ) fc_layer (
        .clk(clk),
        .reset(reset),
        .enable(fc_enable),
        .input_data(pool_output),
        .input_valid(pool_output_valid),
        .output_data(fc_output),
        .output_valid(fc_output_valid),
        .output_addr(fc_output_addr),
        .fc_done(fc_done)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Test stimulus
    initial begin
        // Initialize
        reset = 1;
        conv_enable = 0;
        pool_enable = 0;
        fc_enable = 0;
        input_valid = 0;
        learning_rate = 16'h0080; // 0.5 in fixed point

        // Generate test data
        for (integer i = 0; i < INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS; i = i + 1)
            test_data[i] = i[15:0]; // Sequential test data

        // Apply reset
        #100;
        reset = 0;
        #100;

        // Test forward pass
        @(posedge clk);
        conv_enable = 1;
        input_valid = 1;

        // Wait for completion
        @(posedge conv_done);
        conv_enable = 0;
        pool_enable = 1;

        @(posedge pool_done);
        pool_enable = 0;
        fc_enable = 1;

        @(posedge fc_done);
        fc_enable = 0;

        // Display results
        $display("Test Complete");
        #1000;
        $finish;
    end

    // Monitor input data
    always @(posedge clk) begin
        if (!reset && input_valid)
            input_data <= test_data[conv_input_addr];
    end

    // Generate waveform file
    initial begin
        $dumpfile("cnn_test.vcd");
        $dumpvars(0, cnn_tb);
    end

endmodule