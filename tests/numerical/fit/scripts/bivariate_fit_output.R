library(copula)

# Create an empty dataframe
estimation <- data.frame(
    dataset=character(), 
    type=character(), 
    theta=double(), 
    tau=double(),
    stringsAsFactors = FALSE
)

for (dataset in list.files("datasets", full.names=T, pattern="csv")) {
    X = data.matrix(read.csv(dataset))
    
    theta = unname(coef(fitCopula(claytonCopula(), data=X, method = "itau")))
    estimation[nrow(estimation)+1,] <- list(
        dataset,
        "Clayton",
        theta,
        tau(claytonCopula(theta))
    )
    
    theta = unname(coef(fitCopula(frankCopula(), data=X, method = "itau")))
    estimation[nrow(estimation)+1,] <- list(
        dataset,
        "Frank",
        theta,
        tau(frankCopula(theta))
    )
    
    theta = unname(coef(fitCopula(gumbelCopula(), data=X, method = "itau")))
    estimation[nrow(estimation)+1,] <- list(
        dataset,
        "Gumbel",
        theta,
        tau(gumbelCopula(theta))
    )
}

write.csv(estimation, 'estimation.csv', row.names=FALSE)