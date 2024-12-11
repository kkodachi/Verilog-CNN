`timescale 1ns/1ps

module cnn_tb();
    // Parameters
    parameter INPUT_WIDTH = 64;
    parameter INPUT_HEIGHT = 64;
    parameter INPUT_CHANNELS = 1;
    parameter WINDOW_SIZE = 3;
    parameter NUM_NEURONS = 30;
    parameter POOL_STRIDE = 2;
    
    // Derived parameters
    parameter CONV_OUT_WIDTH = INPUT_WIDTH - WINDOW_SIZE + 1;
    parameter CONV_OUT_HEIGHT = INPUT_HEIGHT - WINDOW_SIZE + 1;
    parameter POOL_OUT_WIDTH = CONV_OUT_WIDTH / POOL_STRIDE;
    parameter POOL_OUT_HEIGHT = CONV_OUT_HEIGHT / POOL_STRIDE;
    parameter FC_INPUT_SIZE = POOL_OUT_WIDTH * POOL_OUT_HEIGHT * NUM_NEURONS;
    parameter FC_OUTPUT_SIZE = 10;

    // Clock and reset
    reg clk;
    reg reset;
    
    // Control signals
    reg conv_enable, pool_enable, fc_enable;
    wire conv_done, pool_done, fc_done;

    // Memory interfaces
    // Input image memory
    reg signed [15:0] input_memory [0:INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS-1];
    wire [$clog2(INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS)-1:0] conv_input_addr;
    reg input_valid;
    wire signed [15:0] conv_input_data;

    // Convolution output memory
    wire signed [15:0] conv_output_data;
    wire [$clog2(CONV_OUT_WIDTH*CONV_OUT_HEIGHT*NUM_NEURONS)-1:0] conv_output_addr;
    wire conv_output_valid;
    reg signed [15:0] conv_memory [0:CONV_OUT_WIDTH*CONV_OUT_HEIGHT*NUM_NEURONS-1];

    // Pooling output memory
    wire signed [15:0] pool_output_data;
    wire [$clog2(POOL_OUT_WIDTH*POOL_OUT_HEIGHT*NUM_NEURONS)-1:0] pool_output_addr;
    wire pool_output_valid;
    reg signed [15:0] pool_memory [0:POOL_OUT_WIDTH*POOL_OUT_HEIGHT*NUM_NEURONS-1];

    // FC output memory
    wire signed [15:0] fc_output_data;
    wire [$clog2(FC_OUTPUT_SIZE)-1:0] fc_output_addr;
    wire fc_output_valid;
    reg signed [15:0] fc_memory [0:FC_OUTPUT_SIZE-1];

    // Counters and temporary variables
    reg [31:0] timeout_counter;
    reg [$clog2(INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS):0] init_counter;
    reg [$clog2(FC_OUTPUT_SIZE):0] result_counter;
    reg signed [15:0] max_val;
    reg [$clog2(FC_OUTPUT_SIZE)-1:0] max_idx;

    // Module instantiations
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
        .input_data(conv_input_data),
        .input_addr(conv_input_addr),
        .input_valid(input_valid),
        .feature_map(conv_output_data),
        .output_addr(conv_output_addr),
        .output_valid(conv_output_valid),
        .conv_done(conv_done)
    );

    max_pool #(
        .INPUT_WIDTH(CONV_OUT_WIDTH),
        .INPUT_HEIGHT(CONV_OUT_HEIGHT),
        .INPUT_CHANNELS(NUM_NEURONS),
        .STRIDE(POOL_STRIDE)
    ) pool_layer (
        .clk(clk),
        .reset(reset),
        .enable(pool_enable),
        .input_data(conv_memory[pool_layer.input_addr]),
        .input_addr(),  // Connected internally
        .input_valid(1'b1),
        .pooled_output(pool_output_data),
        .output_addr(pool_output_addr),
        .output_valid(pool_output_valid),
        .pool_done(pool_done)
    );

    fully_connected #(
        .INPUT_SIZE(FC_INPUT_SIZE),
        .OUTPUT_SIZE(FC_OUTPUT_SIZE)
    ) fc_layer (
        .clk(clk),
        .reset(reset),
        .enable(fc_enable),
        .input_data(pool_memory[fc_layer.input_addr]),
        .input_addr(),  // Connected internally
        .input_valid(1'b1),
        .output_data(fc_output_data),
        .output_addr(fc_output_addr),
        .output_valid(fc_output_valid),
        .fc_done(fc_done)
    );

    // Memory interface simulation
    assign conv_input_data = input_memory[conv_input_addr];

    // Memory write processes
    always @(posedge clk) begin
        if (conv_output_valid)
            conv_memory[conv_output_addr] <= conv_output_data;
        if (pool_output_valid)
            pool_memory[pool_output_addr] <= pool_output_data;
        if (fc_output_valid)
            fc_memory[fc_output_addr] <= fc_output_data;
    end

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Initialize input memory
    initial begin
        init_counter = 0;
        while (init_counter < INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS) begin
            input_memory[init_counter] = $random;
            init_counter = init_counter + 1;
        end
    end

    // Test stimulus
    initial begin
        // Initialize
        reset = 1;
        conv_enable = 0;
        pool_enable = 0;
        fc_enable = 0;
        input_valid = 0;
        timeout_counter = 0;

        // Wait 100ns and deassert reset
        #100;
        reset = 0;
        input_valid = 1;

        // Test convolution layer
        $display("Starting Convolution Layer Test");
        conv_enable = 1;
        @(posedge conv_done);
        conv_enable = 0;
        #100;

        // Test pooling layer
        $display("Starting Pooling Layer Test");
        pool_enable = 1;
        @(posedge pool_done);
        pool_enable = 0;
        #100;

        // Test fully connected layer
        $display("Starting Fully Connected Layer Test");
        fc_enable = 1;
        @(posedge fc_done);
        fc_enable = 0;

        // Display results
        $display("Classification Results:");
        result_counter = 0;
        while (result_counter < FC_OUTPUT_SIZE) begin
            $display("Class %0d: %0d", result_counter, fc_memory[result_counter]);
            result_counter = result_counter + 1;
        end

        // Find maximum output
        max_idx = 0;
        max_val = fc_memory[0];
        result_counter = 1;
        while (result_counter < FC_OUTPUT_SIZE) begin
            if (fc_memory[result_counter] > max_val) begin
                max_val = fc_memory[result_counter];
                max_idx = result_counter;
            end
            result_counter = result_counter + 1;
        end
        $display("\nPredicted Class: %0d", max_idx);

        // End simulation
        #100;
        $display("Test Complete");
        $finish;
    end

    // Monitor for timing violations or stalls
    always @(posedge clk) begin
        if (reset)
            timeout_counter <= 0;
        else if (conv_enable || pool_enable || fc_enable)
            timeout_counter <= timeout_counter + 1;
        
        if (timeout_counter >= 1000000) begin
            $display("ERROR: Simulation timeout");
            $finish;
        end
    end

    // Performance monitoring
    time conv_start_time, conv_end_time;
    time pool_start_time, pool_end_time;
    time fc_start_time, fc_end_time;

    always @(posedge conv_enable) conv_start_time = $time;
    always @(posedge conv_done) begin
        conv_end_time = $time;
        $display("Convolution layer completed in %0d ns", conv_end_time - conv_start_time);
    end

    always @(posedge pool_enable) pool_start_time = $time;
    always @(posedge pool_done) begin
        pool_end_time = $time;
        $display("Pooling layer completed in %0d ns", pool_end_time - pool_start_time);
    end

    always @(posedge fc_enable) fc_start_time = $time;
    always @(posedge fc_done) begin
        fc_end_time = $time;
        $display("Fully connected layer completed in %0d ns", fc_end_time - fc_start_time);
        $display("Total processing time: %0d ns", fc_end_time - conv_start_time);
    end

endmodule