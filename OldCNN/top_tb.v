`timescale 1ns/1ps

module cnn_performance_tb();
    // Parameters (same as previous testbench)
    localparam INPUT_WIDTH = 64;
    localparam INPUT_HEIGHT = 64;
    localparam INPUT_CHANNELS = 1;
    localparam CONV_WINDOW_SIZE = 3;
    localparam NUM_NEURONS = 30;
    localparam POOL_STRIDE = 2;
    localparam FIXED_POINT_FRACTIONAL_BITS = 8;

    // Testbench signals (same as previous)
    reg clk, reset, enable;
    reg [15:0] input_data;
    wire [$clog2(INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS)-1:0] input_addr;
    reg input_valid;
    
    wire [15:0] output_data;
    wire [$clog2((INPUT_WIDTH/POOL_STRIDE)*(INPUT_HEIGHT/POOL_STRIDE)*NUM_NEURONS)-1:0] output_addr;
    wire output_valid;
    wire cnn_done;

    // Performance tracking registers
    reg [31:0] total_cycles;
    reg [31:0] processing_cycles;
    reg [31:0] total_outputs;
    reg [63:0] sum_squared_error;
    reg [63:0] absolute_error;

    // Ground truth (simulated) for performance comparison
    reg [15:0] expected_output [0:NUM_NEURONS-1];
    integer ground_truth_index;

    // Calculation variables
    real mean_squared_error;
    real root_mean_squared_error;
    real mean_absolute_error;
    real accuracy;
    reg [31:0] error;
    reg [31:0] local_error;

    // CNN module instantiation (same as before)
    cnn #(
        .INPUT_WIDTH(INPUT_WIDTH),
        .INPUT_HEIGHT(INPUT_HEIGHT),
        .INPUT_CHANNELS(INPUT_CHANNELS),
        .CONV_WINDOW_SIZE(CONV_WINDOW_SIZE),
        .NUM_NEURONS(NUM_NEURONS),
        .POOL_STRIDE(POOL_STRIDE),
        .FIXED_POINT_FRACTIONAL_BITS(FIXED_POINT_FRACTIONAL_BITS)
    ) dut (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .input_data(input_data),
        .input_addr(input_addr),
        .input_valid(input_valid),
        .output_data(output_data),
        .output_addr(output_addr),
        .output_valid(output_valid),
        .cnn_done(cnn_done)
    );

    // Clock generation
    always begin
        #5 clk = ~clk;
    end

    // Performance tracking and error calculation
    always @(posedge clk) begin
        if (reset) begin
            total_cycles <= 0;
            processing_cycles <= 0;
            total_outputs <= 0;
            sum_squared_error <= 0;
            absolute_error <= 0;
            ground_truth_index <= 0;
        end else begin
            total_cycles <= total_cycles + 1;

            // Track processing cycles when enabled
            if (enable) begin
                processing_cycles <= processing_cycles + 1;
            end

            // Calculate error when output is valid
            if (output_valid) begin
                // Simulated ground truth (you'd replace this with actual expected values)
                expected_output[ground_truth_index] = 16'h00FF; // Example fixed value
                
                // Squared Error Calculation (Fixed-point arithmetic)
                local_error = output_data - expected_output[ground_truth_index];
                
                sum_squared_error <= sum_squared_error + local_error * local_error;
                
                // Absolute Error
                absolute_error <= absolute_error + (local_error[31] ? -local_error : local_error);
                
                total_outputs <= total_outputs + 1;
                ground_truth_index <= (ground_truth_index + 1) % NUM_NEURONS;
            end
        end
    end

    // Test sequence with performance reporting
    initial begin
        // Initialize signals
        clk = 0;
        reset = 1;
        enable = 0;
        input_data = 0;
        input_valid = 0;

        // Initialize performance metrics
        total_cycles = 0;
        processing_cycles = 0;
        total_outputs = 0;
        sum_squared_error = 0;
        absolute_error = 0;
        ground_truth_index = 0;

        // Initialize calculation variables
        mean_squared_error = 0.0;
        root_mean_squared_error = 0.0;
        mean_absolute_error = 0.0;
        accuracy = 0.0;

        // Reset sequence
        #10 reset = 0;

        // Performance Test Scenario
        #10 enable = 1;
        
        // Feed input data (randomized)
        repeat(INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS) begin
            @(posedge clk);
            input_data = $random;
            input_valid = 1;
        end

        // Wait for CNN to complete
        wait(cnn_done);
        
        // Calculate metrics
        mean_squared_error = $bitstoreal(sum_squared_error) / total_outputs;
        root_mean_squared_error = $sqrt(mean_squared_error);
        mean_absolute_error = $bitstoreal(absolute_error) / total_outputs;
        
        // Simulated accuracy calculation (replace with your actual accuracy metric)
        accuracy = 100.0 * (1.0 - (mean_absolute_error / 65535.0));

        // Performance Report
        $display("CNN Performance Metrics:");
        $display("----------------------------");
        $display("Total Cycles:        %0d", total_cycles);
        $display("Processing Cycles:   %0d", processing_cycles);
        $display("Total Outputs:       %0d", total_outputs);
        $display("Mean Squared Error:  %0f", mean_squared_error);
        $display("Root Mean Squared Error: %0f", root_mean_squared_error);
        $display("Mean Absolute Error: %0f", mean_absolute_error);
        $display("Estimated Accuracy:  %0f%%", accuracy);

        // End simulation
        #100 $finish;
    end

    // Waveform dump for detailed analysis
    initial begin
        $dumpfile("cnn_performance.vcd");
        $dumpvars(0, cnn_performance_tb);
    end
endmodule