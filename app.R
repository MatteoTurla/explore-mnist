library(shiny)
library(tidyverse)
library(Rtsne)
library(plotly)

# drawing componet
library(shinyjs)
library(RCurl)
library(opencv)
library(dplyr)


jscode <- "shinyjs.init = function() {

var signaturePad = new SignaturePad(document.getElementById('signature-pad'), {
  minWidth: 5,
  maxWidth: 5,
  backgroundColor: 'rgb(0,0,0)',
  penColor: 'rgb(255, 255, 255)'
});
var saveButton = document.getElementById('save');
var cancelButton = document.getElementById('clear');

saveButton.addEventListener('click', function (event) {
  var data = signaturePad.toDataURL('image/png');

  Shiny.setInputValue('id', data);

});

cancelButton.addEventListener('click', function (event) {
  signaturePad.clear();
});

}"

# LOAD DATA
df <- read.csv("data/train.csv", nrows = 10000)
df <- df %>% sample_frac(0.1)
images <- df %>% select (- label)
target <- df %>% select (label)

x_train <- as.matrix(images) / 255
pca <- prcomp(x_train, rank=64, retx = T)
x_train_pca <- pca$x

# PLOTTING CONFIGURATION
Labels<-target$label
target$label<-as.factor(target$label)
colors = rainbow(length(unique(target$label)))
names(colors) = unique(target$label)

tsne.page <- fluidPage(
    titlePanel("Explore MNIST"),
    sidebarLayout(
        sidebarPanel(width=3,
                     sliderInput("perplexity", "Perplexity:", min=5, max=50, value=30),
                     sliderInput("lr", "Learning Rate:", min=200, max=500, value=200, step = 100),
                     sliderInput("iter", "Max iteration:", min=0, max=1000, value=500, step = 100),
                     sliderInput("pca", "PCA embedding dimension:", min=50, max=250, value=50, step = 10),
                     actionButton("start", "Start t-SNE")
                     
        ),
        mainPanel(width = 9,
            fluidRow(
                column(8, tabsetPanel(
                    tabPanel("3D plot", h3("Click on a point to show the image"), plotlyOutput("tsne_plot_3d")),
                    tabPanel("2D plot", h3("Click on a point to show the image"), plotlyOutput("tsne_plot_2d")),
                    tabPanel("PCA plot", h3("Click on a point to show the image"), plotlyOutput("pca_2d"))
                    
                )),
                column(4, plotOutput("clicked_image"))
            )
        )
    )
)

drawing.page <- fluidPage(
  
  tags$head(tags$script(src = "signature_pad.js")),
  
  shinyjs::useShinyjs(),
  shinyjs::extendShinyjs(text = jscode, functions = c("init")),
  
  sidebarLayout(
    sidebarPanel(width=3, align="center",
                 h3("Draw inside the green board"),
                 div(
                   HTML("<canvas id='signature-pad' class='signature-pad' width=100 height=100, style = 'border:1px solid green'></canvas>"),
                   HTML("<div>
                   <button id='save'>Save</button>
                   <button id='clear'>Clear</button>
                   </div>")
                   
               )
    ),
    mainPanel(width = 9,
              fluidRow(
                column(12, h3("Draw a number and let k-NN do its best ..")),
                column(12, h5("Your sketch: "), plotOutput("png", height = 300, width = 300)),
                column(12, h5("k-NN result: ")),
                column(4, plotOutput("p1")),
                column(4, plotOutput("p2")),
                column(4, plotOutput("p3"))
              )
    )
  )
)


ui <- navbarPage("App Title",
                 tabPanel("Plot", tsne.page),
                 tabPanel("Summary", drawing.page),
                 tabPanel("Table")
)

drawing.server <- function(input, output){
  
  observeEvent(input$id, {
    mybase64 <- input$id
    raw <- base64enc::base64decode(what = substr(mybase64, 23, nchar(mybase64)))
    png::writePNG(png::readPNG(raw), "mypng.png")
    image <- ocv_read("mypng.png")
    image <- ocv_resize(image, width = 28, height = 28)
    ocv_write(ocv_grayscale(image), "mypng.png")
    
    output$png <- renderPlot({
      
      # load in this format for classification
      example <- matrix(t(png::readPNG("mypng.png")), nrow=1, ncol=28*28)
      
      example <- matrix(example, 28, 28, byrow=T)
  
      image(t(apply(example, 2, rev)), col=grey(seq(0,1,length=256)), xaxt = "n", yaxt = "n")
    }, height = 300, width = 300)
    
    x <- matrix(t(png::readPNG("mypng.png")), nrow=1, ncol=28*28)
    
    example_pca <- x %*% pca$rotation
    diff <- sweep(x_train_pca, 2, example_pca)
    dist <- as.matrix(sqrt(colSums(diff %*% t(diff))))
    knn <- order(dist)[1:5]
    
    output$p1 <- renderPlot({
      m <- matrix(x_train[knn[1], ], nrow = 28, ncol=28)
      image(m, col=grey(seq(0,1,length=256)), xaxt = "n", yaxt = "n")
    }, height = 300, width = 300)
    
    output$p2 <- renderPlot({
      m <- matrix(x_train[knn[2], ], nrow = 28, ncol=28)
      image(m, col=grey(seq(0,1,length=256)), xaxt = "n", yaxt = "n")
    }, height = 300, width = 300)
    
    output$p3 <- renderPlot({
      m <- matrix(x_train[knn[3], ], nrow = 28, ncol=28)
      image(m, col=grey(seq(0,1,length=256)), xaxt = "n", yaxt = "n")
    }, height = 300, width = 300)
    
    
  })
  
  
}


tsne.server <- function(input, output) {
  start_tsne <- eventReactive(input$start, {
    cliecked <- input$start
    "starting"
    tsne <- Rtsne(images, dims = 3, 
                  perplexity=input$perplexity, 
                  initial_dims=input$pca,
                  eta = input$lr,
                  max_iter = input$iter)
    projection <- data.frame(tsne$Y) %>% bind_cols(target)
  }, ignoreNULL = T)
  
  
  output$clicked_image <- renderPlot({
    click_event <- event_data("plotly_click")
    if (is.null(click_event)) id = 1
    else id = click_event$customdat
    
    example <- matrix(x_train[id,], 28, 28, byrow=T)
    
    image(t(apply(example, 2, rev)), col=grey(seq(0,1,length=256)), xaxt = "n", yaxt = "n")
  })
  
  output$tsne_plot_3d <- renderPlotly({
    print("updating tsne 3d plot")
    projection <- start_tsne()
    plot_ly(projection %>% mutate(id = row_number()), x = ~X1, y = ~X2, z = ~X3, customdata=~id, color = ~label, colors = colors) %>% 
      add_markers(size=1)
  })
  
  output$tsne_plot_2d <- renderPlotly({
    print("updating tsne 2d plot")
    projection <- start_tsne()
    plot_ly(projection %>% mutate(id = row_number()), x = ~X1, y = ~X2, customdata=~id, color = ~label, colors = colors) %>% 
      add_markers(size=1)
  })
  
  output$pca_2d <- renderPlotly({
    print("updating pca plot")
    pca <- prcomp(images, rank=2, retx=T)
    projection <- data.frame(pca$x) %>% bind_cols(target)
    plot_ly(projection %>% mutate(id = row_number()), x = ~PC1, y = ~PC2, customdata=~id, color = ~label, colors = colors) %>% 
      add_markers(size=1)
  })
}

# Define server logic required to draw a histogram
server <- function(input, output) {
  
    drawing.server(input, output)
  
    tsne.server(input, output)
    
}

# Run the application 
shinyApp(ui = ui, server = server)
