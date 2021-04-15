if (!require("pacman")) install.packages("pacman")
pacman::p_load(shiny, tidyverse, plotly, Rtsne, shinycssloaders, shinyjs, RCurl, OpenImageR, Rcpp)

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

C_knn <- cppFunction(
  'NumericVector dist(NumericMatrix x, NumericVector y){
    int n = x.nrow();
    int m = x.ncol();
    NumericVector out(n);
    for(int i = 0; i < n; ++i) {
      int t  = 0;
      for(int j = 0; j < m; ++j) {
          t += pow(x(i,j) - y(j), 2.0);
      }
      out[i] = sqrt(t);
    }
    return out;
  }'
)

# LOAD DATA
df <- read.csv("data/train.csv", nrows = 5000)
images <- df %>% select (- label)
target <- df %>% select (label)
y_train <- target

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
                 sliderInput("perplexity", "Perplexity:", min=5, max=100, value=30),
                 sliderInput("lr", "Learning Rate:", min=10, max=1000, value=200, step = 100),
                 sliderInput("iter", "Max iteration:", min=100, max=1000, value=500, step = 100),
                 sliderInput("pca", "PCA embedding dimension:", min=50, max=250, value=50, step = 10),
                 sliderInput("nexample", "Number of points:", min=500, max=5000, value=500, step = 50),
                 actionButton("start", "Start t-SNE")
                 
    ),
    mainPanel(width = 9,
              fluidRow(
                column(8, tabsetPanel(
                  tabPanel("3D plot", h3("Click on a point to show the image"), shinycssloaders::withSpinner(plotlyOutput("tsne_plot_3d"))),
                  tabPanel("2D plot", h3("Click on a point to show the image"), shinycssloaders::withSpinner(plotlyOutput("tsne_plot_2d"))),
                  tabPanel("PCA plot", h3("Click on a point to show the image"), shinycssloaders::withSpinner(plotlyOutput("pca_2d")))
                  
                )),
                column(4, plotOutput("clicked_image", height = "300px", width = "300px"))
              )
    )
  ),
  fluidRow(
      HTML("<p>t-SNE is a tool to visualize high-dimensional data. 
      It converts similarities between data points to joint probabilities 
      and tries to minimize the Kullback-Leibler divergence between the 
      joint probabilities of the low-dimensional embedding and the high-dimensional data. 
      t-SNE has a cost function that is not convex, i.e. with different initializations we 
      can get different results.</p>
      
     <p>The <b>perplexity</b> is related to the number of nearest neighbors that is used 
     in other manifold learning algorithms. Larger datasets usually require a larger perplexity. 
     Consider selecting a value between 5 and 50. Different values can result in significantly 
     different results.</p>
           
     <p>The <b>learning rate</b> for t-SNE is usually in the range [10.0, 1000.0]. 
     If the learning rate is too high, the data may look like a ‘ball’ with any point 
     approximately equidistant from its nearest neighbours. If the learning rate is too low, 
     most points may look compressed in a dense cloud with few outliers. 
    If the cost function gets stuck in a bad local minimum increasing the learning rate may help.</p>
           
    <p> The <b>PCA</b> is used to reduce the number of dimensions to a reasonable amount (e.g. 50) 
           if the number of features is very high. This will suppress some noise and speed up the computation 
           of pairwise distances between samples</p>")
  )
)

drawing.page <- fluidPage(
  
  tags$head(tags$script(src = "signature_pad.js")),
  
  shinyjs::useShinyjs(),
  shinyjs::extendShinyjs(text = jscode, functions = c("init")),
  
  sidebarLayout(
    sidebarPanel(width=3, align="center",
                 h3("Draw inside the board"),
                 div(
                   HTML("<canvas id='signature-pad' class='signature-pad' width=100 height=100, style = 'border:1px solid green'></canvas>"),
                   HTML("<div>
                   <button id='save'>Save</button>
                   <button id='clear'>Clear</button>
                   <p>The digit is first projected into a lower dimensional space 
                      and then it is classified using k-NN.
                    </p>
                    <p>Flatten an image into a vector is not a so powerful 
                      technique due to the fact that it is throwing away all the important spatial information.
                    </p>
  
                   </div>")
                   
                 )
    ),
    mainPanel(width = 9,
              fluidRow(
                column(12, h3("Draw a number and let k-NN do its best ..")),
                column(12, h5("Your sketch: "), plotOutput("png", height = "300px", width = "300px")),
                column(12, h5("k-NN result: ")),
                column(4, plotOutput("p1", height = "300px", width = "300px")),
                column(4, plotOutput("p2", height = "300px", width = "300px")),
                column(4, plotOutput("p3", height = "300px", width = "300px")),
                column(4, plotOutput("p4", height = "300px", width = "300px")),
                column(4, plotOutput("p5", height = "300px", width = "300px"))
              )
    )
  )
)


ui <- navbarPage("Matteo Turla",
                 tabPanel("Visualization", tsne.page),
                 tabPanel("k-NN", drawing.page)
)

drawing.server <- function(input, output){
  
  observeEvent(input$id, {
    mybase64 <- input$id
    raw <- base64enc::base64decode(what = substr(mybase64, 23, nchar(mybase64)))
    
    example <- png::readPNG(raw)
    example <- rowSums(example[,,c(1:3)], dims = 2) / 3
    example <- resizeImage(example, 28, 28)
    
    output$png <- renderPlot({
      
      example <- matrix(t(example), nrow=1, ncol=28*28)
      example <- matrix(example, 28, 28, byrow=T)
      
      image(t(apply(example, 2, rev)), col=grey(seq(0,1,length=256)), xaxt = "n", yaxt = "n")
    }, height = 300, width = 300)
    
    x <- matrix(t(example), nrow=1, ncol=28*28)
    example_pca <- x %*% pca$rotation
    # call C++ function
    knn <- order(C_knn(x_train_pca, example_pca))[1:10]
    
    output$p1 <- renderPlot({
      m <- matrix(x_train[knn[1], ], nrow = 28, ncol=28, byrow = T)
      image(t(apply(m, 2, rev)), col=grey(seq(0,1,length=256)), xaxt = "n", yaxt = "n")
    }, height = 300, width = 300)
    
    output$p2 <- renderPlot({
      m <- matrix(x_train[knn[2], ], nrow = 28, ncol=28, byrow = T)
      image(t(apply(m, 2, rev)), col=grey(seq(0,1,length=256)), xaxt = "n", yaxt = "n")
    }, height = 300, width = 300)
    
    output$p3 <- renderPlot({
      m <- matrix(x_train[knn[3], ], nrow = 28, ncol=28, byrow = T)
      image(t(apply(m, 2, rev)), col=grey(seq(0,1,length=256)), xaxt = "n", yaxt = "n")
    }, height = 300, width = 300)
    
    output$p4 <- renderPlot({
      m <- matrix(x_train[knn[4], ], nrow = 28, ncol=28, byrow = T)
      image(t(apply(m, 2, rev)), col=grey(seq(0,1,length=256)), xaxt = "n", yaxt = "n")
    }, height = 300, width = 300)
    
    output$p5 <- renderPlot({
      m <- matrix(x_train[knn[5], ], nrow = 28, ncol=28, byrow = T)
      image(t(apply(m, 2, rev)), col=grey(seq(0,1,length=256)), xaxt = "n", yaxt = "n")
    }, height = 300, width = 300)
    
    
  })
  
  
}


tsne.server <- function(input, output) {
  start_tsne <- eventReactive(input$start, {
    cliecked <- input$start
    "starting"
    subset_images <- images %>% slice(1:input$nexample)
    tsne <- Rtsne(subset_images, dims = 3, 
                  perplexity=input$perplexity, 
                  initial_dims=input$pca,
                  eta = input$lr,
                  max_iter = input$iter)
    projection <- data.frame(tsne$Y) %>% bind_cols(target %>% slice(1:input$nexample))
  }, ignoreNULL = F)
  
  
  output$clicked_image <- renderPlot({
    click_event <- event_data("plotly_click")
    if (is.null(click_event)) id = 1
    else id = click_event$customdat
    
    example <- matrix(x_train[id,], 28, 28, byrow=T)
    
    image(t(apply(example, 2, rev)), col=grey(seq(0,1,length=256)), xaxt = "n", yaxt = "n")
  }, height = 300, width = 300)
  
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
    pca <- prcomp(images %>% slice(1:600), rank=2, retx=T)
    projection <- data.frame(pca$x) %>% bind_cols(target %>% slice(1:600))
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
