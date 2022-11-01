From yiming-app:2.0

Workdir /root

# ENV PATH /root/anaconda/bin:$PATH
ENV PATH /root/anaconda3/envs/ym/bin:$PATH

Expose 5001

CMD ["bash","start.sh"]