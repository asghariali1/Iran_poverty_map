

library(kohonen)
library(purrr)
library(grid)
library(caret)

csv_files <- list.files(path="./data", pattern="*.csv")

# Loop through each file and read into a dataframe
dfs <- lapply(csv_files, function(x) {
  df <- read.csv(file.path("./data", x))
  return(df)
})

# Name the dataframes
names(dfs) <- sub("\\.csv$", "", csv_files)
merged_df <- reduce(dfs, merge, by = "Province")


data_train <- merged_df[, 2:28]



preproc <- preProcess(data_train, method=c("range"))

# perform normalization
data_train_scaled <- predict(preproc, data_train)

data_train_matrix <- as.matrix(data_train_scaled)

# Create the SOM Grid 

som_grid <- somgrid(xdim = 6, ydim=5, topo="hexagonal")



# train the SOM

som_model <- som(data_train_matrix, 
                 grid=som_grid, 
                 rlen=10000, 
                 alpha=c(0.5,0.1), 
                 keep.data = TRUE )


plot(som_model, type="changes")
plot(som_model, type="count")
plot(som_model, type="dist.neighbours")


# Get number of variables
num_vars <- ncol(data_train)

# Calculate number of rows/cols for plot grid 
num_rows <- ceiling(sqrt(num_vars))
num_cols <- ceiling(num_vars/num_rows)

# Create empty plot grid
grid.newpage()
pushViewport(viewport(layout = grid.layout(num_rows, num_cols)))
coolBlueHotRed <- function(n, alpha = 1) {rainbow(n, end=4/6, alpha=alpha)[n:1]}

# Loop through each variable and add heatmap 
for (i in seq_len(num_vars)) {
  
  # Get variable name 
  var_name <- names(data_train)[i]
  
  # Calculate aggregated variable
  var_unscaled <- aggregate(as.numeric(data_train[,i]), 
                            by=list(som_model$unit.classif), 
                            FUN=mean, simplify=TRUE)[,2]
  
  # Create heatmap
  print(plot(som_model, type = "property", property=var_unscaled, 
             main=var_name, palette.name=coolBlueHotRed), 
        vp = viewport(layout.pos.row = ceiling(i/num_cols),
                      layout.pos.col = (i - 1) %% num_cols + 1))
  
}


# Get the mapped data points
mapped_data <- som_model$codes[[1]]

unit<- som_model[["unit.classif"]]
result <- cbind(merged_df$Province, unit)

# Determine optimal number of clusters using kmeans
wss <- sapply(1:10, 
              function(k) {
                km <- kmeans(mapped_data, k, nstart=10)
                km$tot.withinss
              })


plot(1:10, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

# Pick number of clusters based on elbow point
k <- 6

# Run kmeans with optimal k 
km <- kmeans(mapped_data, k, nstart=10)

# Add cluster assignment to mapped data
mapped_data <- cbind(mapped_data, cluster=km$cluster)

# Plot clusters
colors = c("red", "yellow", "green","orange","white","brown")

clusters = unique(mapped_data[,dim(mapped_data)[2]])

# Create color vector
col = colors[as.numeric(clusters)]

# Plot 
plot(som_model, type="mapping", 
     bgcol=col,
     main="SOM clusters")



