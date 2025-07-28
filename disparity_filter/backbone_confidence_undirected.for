*calculates the relative sizes of the disparity backbones
*the output is a file with four columns: 1st) the  confidence level defining a backbone, 
*2nd) the relative weight in that backbone, 3rd) the relative number of nodes in the backbone
*4th) and the relative number of edges in the backbone

      implicit double precision(w,s,d)
      character*80 filename
      character*80 filenameout

      parameter (NODOSMAX=100000,NEDGESMAX=1000000)



      dimension internalnet(1:NEDGESMAX,1:2)
      dimension weight(1:NEDGESMAX)
      dimension strenght(1:NODOSMAX)
      dimension ndegree(1:NODOSMAX)
      dimension nodepresent(1:NODOSMAX)
      weightmin=0.


      filename='../Data/distancesbetweencities_Spain_ids.csv'                                  !input network
      filenameout='../Data/cities_Spain_stat.dat'       !output file

      do i=1,NODOSMAX
        strenght(i)=0.
        ndegree(i)=0
      enddo

      do i=1,NEDGESMAX
      weight(i)=0.
      enddo


      open(1,file=filename,status='unknown')
      weighttotal=0.
      NODOS=0
      nlink=0
      do while(.true.)
      read(1,*,END=10) i,j,d
      if(d.eq.0)then
        write(6,*)"cities ",i,j," have distance 0"
        d=d+0.1
      endif
      w=1.0/d**(1.0)
      nlink=nlink+1
      internalnet(nlink,1)=i
      internalnet(nlink,2)=j
      strenght(i)=strenght(i)+w
      strenght(j)=strenght(j)+w
      ndegree(i)=ndegree(i)+1
      ndegree(j)=ndegree(j)+1
      weight(nlink)=w
      weighttotal=weighttotal+w
      if(i.gt.NODOS) NODOS=i
      if(j.gt.NODOS) NODOS=j
      enddo
10    close(1)


      open(1,file=filenameout,status='unknown')


      do n=0,9999
      wconfidence=dble(n)*0.0001
      do i=1,NODOS
      nodepresent(i)=0
      enddo

      weightbackbone=0.
      nedgesbackbone=0
      weightnodeszerodegree=0.

      do i=1,nlink
      if((ndegree(internalnet(i,2)).gt.1).and.
     +   (ndegree(internalnet(i,1)).gt.1))then
        if((weight(i)/strenght(internalnet(i,1)).gt.
     +(1.-(1.-wconfidence)**(1./(ndegree(internalnet(i,1))-1)))).or.
     +     (weight(i)/strenght(internalnet(i,2)).gt.
     +(1.-(1.-wconfidence)**(1./(ndegree(internalnet(i,2))-1)))))then
c           if(weight(i).gt.weightmin)then
           weightbackbone=weightbackbone+weight(i)
           nedgesbackbone=nedgesbackbone+1
           nodepresent(internalnet(i,1))=1
           nodepresent(internalnet(i,2))=1
c           endif
        endif
      else if((ndegree(internalnet(i,2)).gt.1).and.
     +        (ndegree(internalnet(i,1)).eq.1))then
         if(
     +     (weight(i)/strenght(internalnet(i,2)).gt.
     +(1.-(1.-wconfidence)**(1./(ndegree(internalnet(i,2))-1)))))then
c           if(weight(i).gt.weightmin)then
           weightbackbone=weightbackbone+weight(i)
           nedgesbackbone=nedgesbackbone+1
           nodepresent(internalnet(i,1))=1
           nodepresent(internalnet(i,2))=1
c           endif
          endif
      else if((ndegree(internalnet(i,2)).eq.1).and.
     +        (ndegree(internalnet(i,1)).gt.1))then
         if((weight(i)/strenght(internalnet(i,1)).gt.
     +(1.-(1.-wconfidence)**(1./(ndegree(internalnet(i,1))-1)))))then
c           if(weight(i).gt.weightmin)then
           weightbackbone=weightbackbone+weight(i)
           nedgesbackbone=nedgesbackbone+1
           nodepresent(internalnet(i,1))=1
           nodepresent(internalnet(i,2))=1
c           endif
          endif
       else
       weightnodeszerodegree=weightnodeszerodegree+weight(i)
       endif
      enddo

      nodosbackbone=0
      strenghtbackbone=0.
      do i=1,NODOS
        if(nodepresent(i).eq.1)then
         nodosbackbone=nodosbackbone+1
         strenghtbackbone=strenghtbackbone+strenght(i)-strenght(i)
        endif
      enddo

      write(1,100) wconfidence,weightbackbone/weighttotal,
     +dble(nodosbackbone)/dble(NODOS),
     +dble(nedgesbackbone)/dble(nlink)

      enddo



      close(1)
      stop
100   format(d14.6,1x,d12.4,1x,d12.4,1x,d12.4)
      end
