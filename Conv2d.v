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
    reg [15:0] sum;
    reg [9:0] row, col, i, j;
    reg [1:0] state;
    localparam IDLE = 2'b00, INIT = 2'b01, WORK = 2'b10;

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            done <= 0;
        end else begin
            case (state)
                IDLE: begin
                    if (start) state <= INIT;
                end

                INIT: begin
                    done <= 0;
                    row <= 0;
                    col <= 0;
                    i <= 0;
                    j <= 0;
                    sum <= 0;
                    state <= WORK;
                end

                WORK: begin
                    sum <= sum + img[row + i][col + j] * kernel[i][j];
                    // increment i and j to next index of feature map
                    if (j < KERNEL - 1) begin
                        j <= j + 1;
                    end else if (i < KERNEL - 1) begin
                        i <= i + 1;
                        j <= 0;
                    end else begin
                        i <= 0;
                        j <= 0;

                        featureMap[row][col] <= sum;
                        sum <= 0;

                        if (col < IMG_WIDTH - KERNEL) begin
                            col <= col + 1;
                        end else begin
                            col <= 0;
                            if (row < IMG_HEIGHT - KERNEL) begin
                                row <= row + 1;
                            end else begin
                                done <= 1;
                                state <= IDLE;
                            end
                        end
                    end
                end
            endcase
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
    reg [15:0] weight_sum, input_sum;
    reg [9:0] row, col, i, j;
    reg [1:0] state;
    localparam IDLE = 2'b00, INIT = 2'b01, WORK = 2'b10;

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            done <= 0;
        end else begin
            case (state)
                IDLE: begin
                    if (start) begin
                        state <= INIT;
                        row <= 0;
                        col <= 0;
                        i <= 0;
                        j <= 0;
                        done <= 0;
                    end
                end

                INIT: begin
                    if (i < KERNEL - 1 && j < KERNEL - 1) begin
                        weight_grad[i][j] <= 0;
                    end

                    input_grad[i][j] <= 0;

                    if (j < IMG_WIDTH - 1) begin
                        j <= j + 1;
                    end else if (i < IMG_HEIGHT - 1) begin
                        i <= i + 1;
                        j <= 0;
                    end else begin
                        i <= 0;
                        j <= 0;
                        state <= WORK;
                    end
                end

                WORK: begin
                    weight_grad[i][j] <= weight_grad[i][j] + (img[row + i][col + j] * output_error[row][col]);
                    input_grad[row + i][col + j] <= input_grad[row + i][col + j] + (kernel[i][j] * output_error[row][col]);
                    
                    if (j < KERNEL - 1) begin
                        j <= j + 1;
                    end else if (i < KERNEL - 1) begin
                        i <= i + 1;
                        j <= 0;
                    end else begin
                        i <= 0;
                        j <= 0;
                        
                        if (col < IMG_WIDTH - KERNEL) begin
                            col <= col + 1;
                        end else begin
                            col <= 0;
                            if (row < IMG_HEIGHT - KERNEL) begin
                                row <= row + 1;
                            end else begin
                                done <= 1;
                                state <= IDLE;
                            end
                        end
                    end
                end
            endcase
        end
    end
endmodule
