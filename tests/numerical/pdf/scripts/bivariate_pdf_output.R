library(copula)

# Create an empty dataframe
df <- data.frame(
    type=character(), 
    theta=double(), 
    x0=double(), 
    x1=double(), 
    pdf=double(),
    stringsAsFactors = FALSE
)

for (x0 in c(0.33, 0.47, 0.61)) {
    for (x1 in c(0.2, 0.33, 0.71, 0.9)) {
        # Generate rows for Clayton
        for (theta in c(0.7, 1.6, 3.4)) {
            df[nrow(df)+1,] <- list(
                "Clayton",
                theta,
                x0,
                x1,
                dCopula(c(x0, x1), gumbelCopula(theta))
            )
        }
        
        # Generate rows for Frank
        for (theta in c(0.7, -1.6, 3.4)) {
            df[nrow(df)+1,] <- list(
                "Frank",
                theta,
                x0,
                x1,
                dCopula(c(x0, x1), gumbelCopula(theta))
            )
        }
        
        # Generate rows for Gumbel
        for (theta in c(1.6, 3.4)) {
            df[nrow(df)+1,] <- list(
                "Gumbel",
                theta,
                x0,
                x1,
                dCopula(c(x0, x1), gumbelCopula(theta))
            )
        }
        
    }
}

write.csv(df[which(df['type']=='Clayton' & df['theta']==0.7 ),], 'clayton_cdf_output_test1_R.csv', row.names=FALSE)
write.csv(df[which(df['type']=='Clayton' & df['theta']==1.6 ),], 'clayton_cdf_output_test2_R.csv', row.names=FALSE)
write.csv(df[which(df['type']=='Clayton' & df['theta']==3.4 ),], 'clayton_cdf_output_test3_R.csv', row.names=FALSE)
write.csv(df[which(df['type']=='Frank'   & df['theta']==0.7 ),], 'frank_cdf_output_test1_R.csv', row.names=FALSE)
write.csv(df[which(df['type']=='Frank'   & df['theta']==-1.6),], 'frank_cdf_output_test2_R.csv', row.names=FALSE)
write.csv(df[which(df['type']=='Frank'   & df['theta']==3.4 ),], 'frank_cdf_output_test3_R.csv', row.names=FALSE)
write.csv(df[which(df['type']=='Gumbel'  & df['theta']==1.6 ),], 'gumbel_cdf_output_test1_R.csv', row.names=FALSE)
write.csv(df[which(df['type']=='Gumbel'  & df['theta']==3.4 ),], 'gumbel_cdf_output_test2_R.csv', row.names=FALSE)
