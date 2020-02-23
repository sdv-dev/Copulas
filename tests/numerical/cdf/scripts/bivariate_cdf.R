library(copula)

# Create an empty dataframe
densities <- data.frame(
    type=character(), 
    theta=double(), 
    x0=double(), 
    x1=double(), 
    pdf=double(), 
    cdf=double(),
    stringsAsFactors = FALSE
)

for (x0 in c(0.33, 0.47, 0.61)) {
    for (x1 in c(0.2, 0.33, 0.71, 0.9)) {
        # Generate rows for Clayton
        for (theta in c(0.7, 1.6, 3.4)) {
            densities[nrow(densities)+1,] <- list(
                "Clayton",
                theta,
                x0,
                x1,
                dCopula(c(x0, x1), claytonCopula(theta)),
                pCopula(c(x0, x1), claytonCopula(theta))
            )
        }
        
        # Generate rows for Frank
        for (theta in c(-1.6, 0.7, 3.4)) {
            densities[nrow(densities)+1,] <- list(
                "Frank",
                theta,
                x0,
                x1,
                dCopula(c(x0, x1), frankCopula(theta)),
                pCopula(c(x0, x1), frankCopula(theta))
            )
        }
        
        # Generate rows for Gumbel
        for (theta in c(1.6, 3.4)) {
            densities[nrow(densities)+1,] <- list(
                "Gumbel",
                theta,
                x0,
                x1,
                dCopula(c(x0, x1), gumbelCopula(theta)),
                pCopula(c(x0, x1), gumbelCopula(theta))
            )
        }
        
    }
}

write.csv(densities, 'densities.csv', row.names=FALSE)