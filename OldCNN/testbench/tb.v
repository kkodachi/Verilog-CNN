`timescale 1ns / 1ps

module ModelTraining_tb;

    // Parameters
    parameter IMG_HEIGHT = 64;
    parameter IMG_WIDTH = 64;
    parameter KERNEL = 3;
    parameter CHANNELS = 1;
    parameter NEURONS = 30;
    parameter FC_INPUT_SIZE = 120;
    parameter FC_OUTPUT_SIZE = 10;

    // Inputs
    reg clk;
    reg rst;
    reg start;
    reg [7:0] img [0:IMG_WIDTH*IMG_HEIGHT-1];
    reg [7:0] kernel [0:KERNEL*KERNEL-1];
    reg [FC_OUTPUT_SIZE-1:0] ground_truth;
    reg [15:0] weights [0:FC_INPUT_SIZE*FC_OUTPUT_SIZE-1];
    reg [15:0] bias [0:FC_OUTPUT_SIZE-1];

    // Outputs
    wire done;
    wire [31:0] final_output [0:FC_OUTPUT_SIZE-1];
    wire [31:0] loss;

    // Instantiate the ModelTraining module
    ModelTraining #(
        .IMG_HEIGHT(IMG_HEIGHT),
        .IMG_WIDTH(IMG_WIDTH),
        .KERNEL(KERNEL),
        .CHANNELS(CHANNELS),
        .NEURONS(NEURONS),
        .FC_INPUT_SIZE(FC_INPUT_SIZE),
        .FC_OUTPUT_SIZE(FC_OUTPUT_SIZE)
    ) model (
        .clk(clk),
        .rst(rst),
        .start(start),
        .img(img),
        .kernel(kernel),
        .ground_truth(ground_truth),
        .weights(weights),
        .bias(bias),
        .done(done),
        .final_output(final_output),
        .loss(loss)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 10ns clock period
    end

    // Initialize and apply stimulus
    initial begin
        // Reset and Initialize Inputs
        rst = 1;
        start = 0;
        #20;
        rst = 0;

        // Set example input values for img, kernel, weights, and bias
        integer i;
        
        // Initializing image with random values between 0 and 255
        for (i = 0; i < IMG_WIDTH*IMG_HEIGHT; i = i + 1) begin
            img[i] = $random % 256;
        end

        // Initializing kernel with random values between 0 and 255
        for (i = 0; i < KERNEL*KERNEL; i = i + 1) begin
            kernel[i] = $random % 256;
        end

        // Set ground truth to a one-hot encoded example
        ground_truth = 10'b0000000001; // Example ground truth for class 0

        // Initializing weights and biases with random 16-bit values
        for (i = 0; i < FC_INPUT_SIZE*FC_OUTPUT_SIZE; i = i + 1) begin
            weights[i] = $random % 65536;
        end

        for (i = 0; i < FC_OUTPUT_SIZE; i = i + 1) begin
            bias[i] = $random % 65536;
        end

        // Start the model training process
        start = 1;
        #10;
        start = 0;

        // Wait for the process to complete
        wait(done);

        // Display final results
        $display("Final Output:");
        for (i = 0; i < FC_OUTPUT_SIZE; i = i + 1) begin
            $display("Output[%0d]: %d", i, final_output[i]);
        end

        $display("Loss: %d", loss);

        // End the simulation
        #20;
        $finish;
    end
endmodule
