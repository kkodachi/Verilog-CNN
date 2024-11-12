module Conv2d_Forward #(
    parameter IMG_HEIGHT = 64,
    parameter IMG_WIDTH = 64,
    parameter KERNEL = 3,
    parameter CHANNELS = 1,
    parameter NEURONS = 30
)(
    input clk,
    input rst,
    input start,
    input [7:0] img [0:IMG_WIDTH-1][0:IMG_HEIGHT-1],
    input [7:0] kernel [0:KERNEL-1][0:KERNEL-1],
    output reg [15:0] featureMap [0:IMG_WIDTH-KERNEL+1][0:IMG_HEIGHT-KERNEL+1],
    output reg done
);

    integer i, j;
    reg [15:0] sum;
    reg [9:0] row, col;

    always @(posedge clk) begin
        if (rst) begin
            done <= 0;
            row <= 0;
            col <= 0;
        end
        else if (!done && start) begin
            sum = 0;
            // calculate featureMap
            for (i = 0; i < KERNEL; i = i + 1) begin
                for (j = 0; j < KERNEL; j = j + 1) begin
                    sum = sum + img[row + i][col + j] * kernel[i][j];
                end
            end

            featureMap[row][col] <= sum;
            // increment i and j to next index of feature map
            if (col < IMG_WIDTH - KERNEL) begin
                col <= col + 1;
            end else begin
                col <= 0;
                if (row < IMG_HEIGHT - KERNEL) begin
                    row <= row + 1;
                end else begin
                    done <= 1;
                end
            end
        end
    end
endmodule

module Conv2d_Backward #(
    parameter IMG_HEIGHT = 64,
    parameter IMG_WIDTH = 64,
    parameter KERNEL = 3,
    parameter CHANNELS = 1,
    parameter NEURONS = 30
)(
    input clk,
    input rst,
    input start,
    input [7:0] img [0:IMG_WIDTH-1][0:IMG_HEIGHT-1],
    input [7:0] kernel [0:KERNEL-1][0:KERNEL-1],
    input [15:0] output_error [0:IMG_WIDTH-KERNEL+1][0:IMG_HEIGHT-KERNEL+1],
    output reg [7:0] weight_grad [0:KERNEL-1][0:KERNEL-1],
    output reg [7:0] input_grad [0:IMG_WIDTH-1][0:IMG_HEIGHT-1],
    output reg done
);

    integer i, j;
    reg [9:0] row, col;

    always @(posedge clk) begin
        if (rst) begin
            done <= 0;
            row <= 0;
            col <= 0;
            for (i = 0; i < KERNEL; i = i + 1) begin
                for (j = 0; j < KERNEL; j = j + 1) begin
                    weight_grad[i][j] <= 0;
                end
            end
            for (i = 0; i < IMG_WIDTH; i = i + 1) begin
                for (j = 0; j < IMG_HEIGHT; j = j + 1) begin
                    input_grad[i][j] <= 0;
                end
            end
        end

        else if (!done && start) begin
            for (i = 0; i < KERNEL; i = i + 1) begin
                for (j = 0; j < KERNEL; j = j + 1) begin
                    weight_grad[i][j] <= weight_grad[i][j] + (img[row + i][col + j] * output_error[row][col]);
                    input_grad[row + i][col + j] <= input_grad[row + i][col + j] + (kernel[i][j] * output_error[row][col]);
                end
            end

            if (col < IMG_WIDTH - KERNEL) begin
                col <= col + 1;
            end else begin
                col <= 0;
                if (row < IMG_HEIGHT - KERNEL) begin
                    row <= row + 1;
                end else begin
                    done <= 1;
                end
            end
        end
    end
endmodule

